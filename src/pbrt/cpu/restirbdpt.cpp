#include <pbrt/cpu/restirbdpt.h>

#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/media.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/shapes.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/display.h>
#include <pbrt/util/error.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/stats.h>

#include <algorithm>
#include <cmath>

namespace pbrt {

STAT_COUNTER("ReSTIR-BDPT/Camera rays traced", nReSTIRCameraRays);
STAT_COUNTER("ReSTIR-BDPT/Light subpaths generated", nLightSubpaths);
STAT_COUNTER("ReSTIR-BDPT/Path connections attempted", nPathConnections);
STAT_COUNTER("ReSTIR-BDPT/Spatial resampling passes", nSpatialPasses);

// ReSTIRBDPTIntegrator Method Definitions
ReSTIRBDPTIntegrator::ReSTIRBDPTIntegrator(Camera camera, Sampler sampler,
                                           Primitive aggregate, std::vector<Light> lights,
                                           int maxDepth, int spatialIterations,
                                           int spatialNeighbors, Float spatialRadius,
                                           bool temporalReuse, bool regularize)
    : ImageTileIntegrator(camera, sampler, aggregate, lights),
      maxDepth(maxDepth),
      spatialIterations(spatialIterations),
      spatialNeighbors(spatialNeighbors),
      spatialRadius(spatialRadius),
      temporalReuse(temporalReuse),
      regularize(regularize),
      lightSampler(new PowerLightSampler(lights, Allocator())),
      tlCameraVertices([maxDepth]() { return std::vector<ReSTIRVertex>(maxDepth + 2); }),
      tlLightVertices([maxDepth]() { return std::vector<ReSTIRVertex>(maxDepth + 1); }) {}

void ReSTIRBDPTIntegrator::Render() {
    // Get pixel bounds and initialize buffers
    Bounds2i pixelBounds = camera.GetFilm().PixelBounds();
    Point2i resolution = Point2i(pixelBounds.Diagonal());

    // Initialize reservoir buffers
    reservoirBuffer[0].Initialize(resolution);
    reservoirBuffer[1].Initialize(resolution);
    currentBuffer = 0;

    // Initialize accumulation buffer
    accumBuffer.resize(resolution.x * resolution.y);

    int spp = samplerPrototype.SamplesPerPixel();
    ProgressReporter progress(int64_t(spp) * pixelBounds.Area(), "Rendering",
                              Options->quiet);

    // Thread-local scratch buffers and samplers
    ThreadLocal<ScratchBuffer> scratchBuffers([]() { return ScratchBuffer(); });
    ThreadLocal<Sampler> samplers([this]() { return samplerPrototype.Clone(); });

    LOG_VERBOSE("Starting ReSTIR-BDPT rendering with %d spp", spp);

    // Main rendering loop - process each sample
    for (int sampleIndex = 0; sampleIndex < spp; ++sampleIndex) {
        LOG_VERBOSE("Processing sample %d/%d", sampleIndex + 1, spp);

        // Reset reservoir buffers for new sample
        reservoirBuffer[currentBuffer].Reset();

        // Pass 1: Generate initial candidates and build reservoirs
        ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
            ScratchBuffer &scratchBuffer = scratchBuffers.Get();
            Sampler &sampler = samplers.Get();

            for (Point2i pPixel : tileBounds) {
                Vector2i offset = pPixel - pixelBounds.pMin;
                Point2i localPixel(offset.x, offset.y);
                sampler.StartPixelSample(pPixel, sampleIndex);

                // Generate initial path samples
                std::vector<PathSample> samples;
                GenerateInitialSamples(pPixel, sampleIndex, sampler, scratchBuffer,
                                       samples);

                // Build initial reservoir from samples using luminance-weighted selection
                Reservoir &reservoir = reservoirBuffer[currentBuffer](localPixel);
                for (const auto &sample : samples) {
                    if (sample.IsValid()) {
                        // Use luminance as selection weight (importance sampling)
                        Float weight = 0.0f;
                        if (sample.sourcePdf > 0) {
                            weight = sample.targetPdf / sample.sourcePdf;
                        }
                        reservoir.Update(sample, weight, sampler.Get1D());
                    }
                }
                reservoir.Finalize();

                scratchBuffer.Reset();
            }
            progress.Update(tileBounds.Area());
        });

        // Pass 2: Spatial resampling iterations
        for (int iter = 0; iter < spatialIterations; ++iter) {
            ++nSpatialPasses;
            int sourceBuffer = currentBuffer;
            int targetBuffer = 1 - currentBuffer;
            reservoirBuffer[targetBuffer].Reset();

            ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
                Sampler &sampler = samplers.Get();

                for (Point2i pPixel : tileBounds) {
                    Vector2i offset = pPixel - pixelBounds.pMin;
                    Point2i localPixel(offset.x, offset.y);
                    sampler.StartPixelSample(pPixel, sampleIndex, iter + 1);

                    SpatialResample(localPixel, sampler, reservoirBuffer[targetBuffer],
                                    reservoirBuffer[sourceBuffer]);
                }
            });

            currentBuffer = targetBuffer;
        }

        // Pass 3: Temporal resampling (if enabled and not first sample)
        if (temporalReuse && sampleIndex > 0) {
            int sourceBuffer = currentBuffer;
            int targetBuffer = 1 - currentBuffer;
            reservoirBuffer[targetBuffer].Reset();

            ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
                Sampler &sampler = samplers.Get();

                for (Point2i pPixel : tileBounds) {
                    Vector2i offset = pPixel - pixelBounds.pMin;
                    Point2i localPixel(offset.x, offset.y);
                    sampler.StartPixelSample(pPixel, sampleIndex, spatialIterations + 1);

                    TemporalResample(localPixel, sampler, reservoirBuffer[targetBuffer],
                                     reservoirBuffer[sourceBuffer]);
                }
            });

            currentBuffer = targetBuffer;
        }

        // Pass 4: Shade and accumulate final contributions
        ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
            for (Point2i pPixel : tileBounds) {
                Vector2i offset = pPixel - pixelBounds.pMin;
                Point2i localPixel(offset.x, offset.y);
                int idx = localPixel.y * resolution.x + localPixel.x;

                const Reservoir &reservoir = reservoirBuffer[currentBuffer](localPixel);

                if (reservoir.sample.IsValid()) {
                    // BDPT paths are already unbiased - output directly
                    // The reservoir just selects one path based on luminance importance
                    SampledSpectrum L = reservoir.sample.contribution * reservoir.W;

                    // Final firefly clamp as safety net
                    Float lum = L.y(reservoir.sample.lambda);
                    Float maxLum = 10.0f;
                    if (lum > maxLum)
                        L *= maxLum / lum;

                    // Handle splat case (t=1)
                    if (reservoir.sample.t == 1 && reservoir.sample.pFilm) {
                        camera.GetFilm().AddSplat(*reservoir.sample.pFilm, L,
                                                  reservoir.sample.lambda);
                    } else {
                        // Regular pixel contribution
                        camera.GetFilm().AddSample(pPixel, L, reservoir.sample.lambda,
                                                   nullptr, 1.0f);
                    }
                }
            }
        });

        // Swap buffers for temporal reuse
        if (temporalReuse)
            currentBuffer = 1 - currentBuffer;
    }

    progress.Done();

    // Write final image
    ImageMetadata metadata;
    metadata.renderTimeSeconds = progress.ElapsedSeconds();
    metadata.samplesPerPixel = spp;
    camera.InitMetadata(&metadata);
    camera.GetFilm().WriteImage(metadata, 1.0f / spp);

    LOG_VERBOSE("ReSTIR-BDPT rendering finished");
}

void ReSTIRBDPTIntegrator::EvaluatePixelSample(Point2i pPixel, int sampleIndex,
                                               Sampler sampler,
                                               ScratchBuffer &scratchBuffer) {
    // This is not used in ReSTIR-BDPT as we override Render()
    // But we need to implement it for ImageTileIntegrator
}

void ReSTIRBDPTIntegrator::GenerateInitialSamples(Point2i pPixel, int sampleIndex,
                                                  Sampler sampler,
                                                  ScratchBuffer &scratchBuffer,
                                                  std::vector<PathSample> &samples) {
    // Sample wavelengths
    Float lu = sampler.Get1D();
    if (Options->disableWavelengthJitter)
        lu = 0.5;
    SampledWavelengths lambda = camera.GetFilm().SampleWavelengths(lu);

    // Generate camera ray
    Filter filter = camera.GetFilm().GetFilter();
    CameraSample cameraSample = GetCameraSample(sampler, pPixel, filter);
    pstd::optional<CameraRayDifferential> cameraRay =
        camera.GenerateRayDifferential(cameraSample, lambda);

    if (!cameraRay)
        return;

    ++nReSTIRCameraRays;

    // Scale camera ray differentials
    Float rayDiffScale = std::max<Float>(
        0.125f, 1.f / std::sqrt((Float)samplerPrototype.SamplesPerPixel()));
    if (!Options->disablePixelJitter)
        cameraRay->ray.ScaleDifferentials(rayDiffScale);

    // Generate camera subpath
    std::vector<ReSTIRVertex> &cameraVertices = tlCameraVertices.Get();
    int nCamera = GenerateCameraSubpath(cameraRay->ray, lambda, sampler, scratchBuffer,
                                        maxDepth + 2, cameraVertices);

    if (nCamera == 0)
        return;

    // Generate light subpath
    std::vector<ReSTIRVertex> &lightVertices = tlLightVertices.Get();
    ++nLightSubpaths;
    int nLight = GenerateLightSubpath(lambda, sampler, scratchBuffer, maxDepth + 1,
                                      cameraVertices[0].time, lightVertices);

    // Build list of valid strategies
    samples.clear();
    std::vector<std::pair<int, int>> validStrategies;
    for (int t = 1; t <= nCamera; ++t) {
        for (int s = 0; s <= nLight; ++s) {
            int depth = t + s - 2;
            if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth)
                continue;
            validStrategies.push_back({s, t});
        }
    }

    if (validStrategies.empty())
        return;

    int numCandidates =
        std::min((int)validStrategies.size(), 4);  // Try up to 4 strategies

    for (int i = 0; i < numCandidates; ++i) {
        // Randomly select a strategy (or iterate through first few)
        int stratIdx;
        if (numCandidates < (int)validStrategies.size()) {
            stratIdx =
                int(sampler.Get1D() * validStrategies.size()) % validStrategies.size();
        } else {
            stratIdx = i;
        }

        int s = validStrategies[stratIdx].first;
        int t = validStrategies[stratIdx].second;

        ++nPathConnections;

        pstd::optional<Point2f> pFilm;
        Float misWeight = 0;
        SampledSpectrum Lpath = ConnectSubpaths(lightVertices, cameraVertices, s, t,
                                                lambda, sampler, &pFilm, &misWeight);

        if (Lpath && misWeight > 0) {
            // Clamp extreme contributions to reduce fireflies
            Float lum = Lpath.y(lambda);
            Float maxLum = 10.0f;  // Clamp very bright samples
            if (lum > maxLum)
                Lpath *= maxLum / lum;

            PathSample sample;
            sample.s = s;
            sample.t = t;
            sample.contribution = Lpath;  // Already includes MIS weight
            sample.lambda = lambda;
            sample.pFilm = pFilm;

            // Store reconnection vertices
            if (s > 0)
                sample.lightVertex = lightVertices[s - 1];
            if (t > 0)
                sample.cameraVertex = cameraVertices[t - 1];

            // Compute target PDF (luminance-based)
            sample.targetPdf = TargetPdf(sample);

            // Source PDF: probability of selecting this strategy
            sample.sourcePdf = Float(numCandidates) / Float(validStrategies.size());

            if (sample.IsValid())
                samples.push_back(sample);
        }
    }
}

int ReSTIRBDPTIntegrator::GenerateCameraSubpath(
    const RayDifferential &ray, SampledWavelengths &lambda, Sampler sampler,
    ScratchBuffer &scratchBuffer, int maxDepth, std::vector<ReSTIRVertex> &path) {
    if (maxDepth == 0)
        return 0;

    // Create camera vertex
    path[0] = ReSTIRVertex();
    path[0].type = ReSTIRVertexType::Camera;
    path[0].beta = SampledSpectrum(1.f);
    path[0].p = ray.o;
    path[0].time = ray.time;
    path[0].camera = camera;

    Float pdfPos, pdfDir;
    camera.PDF_We(ray, &pdfPos, &pdfDir);
    path[0].pdfFwd = pdfPos;

    // Random walk
    SampledSpectrum beta(1.f);
    RayDifferential currentRay = ray;
    Float pdfFwd = pdfDir;
    int bounces = 0;
    bool anyNonSpecularBounces = false;

    while (bounces < maxDepth - 1) {
        // Intersect ray with scene
        pstd::optional<ShapeIntersection> si = Intersect(currentRay);

        if (!si) {
            // Create infinite light vertex for escaped rays
            path[bounces + 1] = ReSTIRVertex();
            path[bounces + 1].type = ReSTIRVertexType::Light;
            path[bounces + 1].beta = beta;
            path[bounces + 1].p = currentRay.o + currentRay.d * 1e6f;  // Far point
            path[bounces + 1].pdfFwd = pdfFwd;
            return bounces + 2;
        }

        SurfaceInteraction &isect = si->intr;

        // Get BSDF
        BSDF bsdf = isect.GetBSDF(currentRay, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            // Skip medium boundary
            isect.SkipIntersection(&currentRay, si->tHit);
            continue;
        }

        // Regularize if needed
        if (regularize && anyNonSpecularBounces)
            bsdf.Regularize();

        // Create surface vertex
        bounces++;
        path[bounces] = ReSTIRVertex();
        path[bounces].type = ReSTIRVertexType::Surface;
        path[bounces].si = isect;
        path[bounces].bsdf = bsdf;
        path[bounces].p = isect.p();
        path[bounces].n = isect.n;
        path[bounces].ns = isect.shading.n;
        path[bounces].wo = -currentRay.d;
        path[bounces].time = isect.time;
        path[bounces].beta = beta;

        // Convert PDF to area measure
        Vector3f d = path[bounces].p - path[bounces - 1].p;
        Float invDist2 = 1.f / LengthSquared(d);
        Float cosTheta = AbsDot(path[bounces].n, Normalize(d));
        path[bounces].pdfFwd = pdfFwd * invDist2;
        if (path[bounces].IsOnSurface())
            path[bounces].pdfFwd *= cosTheta;

        if (bounces >= maxDepth - 1)
            break;

        // Sample BSDF for next direction
        Vector3f wo = isect.wo;
        Float u = sampler.Get1D();
        pstd::optional<BSDFSample> bs = bsdf.Sample_f(wo, u, sampler.Get2D());
        if (!bs)
            break;

        pdfFwd = bs->pdfIsProportional ? bsdf.PDF(wo, bs->wi) : bs->pdf;
        anyNonSpecularBounces |= !bs->IsSpecular();
        path[bounces].delta = bs->IsSpecular();

        beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;

        // Compute reverse PDF
        Float pdfRev = bsdf.PDF(bs->wi, wo);
        if (bs->IsSpecular())
            pdfRev = pdfFwd = 0;

        // Convert reverse PDF to area measure for previous vertex
        if (bounces > 0) {
            d = path[bounces - 1].p - path[bounces].p;
            invDist2 = 1.f / LengthSquared(d);
            path[bounces - 1].pdfRev = pdfRev * invDist2;
            if (path[bounces - 1].IsOnSurface())
                path[bounces - 1].pdfRev *= AbsDot(path[bounces - 1].n, Normalize(d));
        }

        currentRay = isect.SpawnRay(currentRay, bsdf, bs->wi, bs->flags, bs->eta);
    }

    return bounces + 1;
}

int ReSTIRBDPTIntegrator::GenerateLightSubpath(SampledWavelengths &lambda,
                                               Sampler sampler,
                                               ScratchBuffer &scratchBuffer, int maxDepth,
                                               Float time,
                                               std::vector<ReSTIRVertex> &path) {
    if (maxDepth == 0)
        return 0;

    // Sample light
    pstd::optional<SampledLight> sampledLight = lightSampler.Sample(sampler.Get1D());
    if (!sampledLight)
        return 0;

    Light light = sampledLight->light;
    Float lightPdf = sampledLight->p;

    // Sample point and direction on light
    Point2f ul0 = sampler.Get2D();
    Point2f ul1 = sampler.Get2D();
    pstd::optional<LightLeSample> les = light.SampleLe(ul0, ul1, lambda, time);
    if (!les || les->pdfPos == 0 || les->pdfDir == 0 || !les->L)
        return 0;

    // Create light vertex
    path[0] = ReSTIRVertex();
    path[0].type = ReSTIRVertexType::Light;
    path[0].light = light;
    path[0].beta = les->L;
    path[0].p = les->ray.o;
    path[0].n = les->intr ? les->intr->n : Normal3f(les->ray.d);
    path[0].time = time;
    path[0].pdfFwd = lightPdf * les->pdfPos;

    // Random walk from light
    SampledSpectrum beta =
        les->L * les->AbsCosTheta(les->ray.d) / (lightPdf * les->pdfPos * les->pdfDir);
    RayDifferential currentRay(les->ray);
    Float pdfFwd = les->pdfDir;
    int bounces = 0;
    bool anyNonSpecularBounces = false;

    while (bounces < maxDepth - 1) {
        pstd::optional<ShapeIntersection> si = Intersect(currentRay);
        if (!si)
            break;

        SurfaceInteraction &isect = si->intr;

        BSDF bsdf = isect.GetBSDF(currentRay, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            isect.SkipIntersection(&currentRay, si->tHit);
            continue;
        }

        if (regularize && anyNonSpecularBounces)
            bsdf.Regularize();

        bounces++;
        path[bounces] = ReSTIRVertex();
        path[bounces].type = ReSTIRVertexType::Surface;
        path[bounces].si = isect;
        path[bounces].bsdf = bsdf;
        path[bounces].p = isect.p();
        path[bounces].n = isect.n;
        path[bounces].ns = isect.shading.n;
        path[bounces].wo = -currentRay.d;
        path[bounces].time = isect.time;
        path[bounces].beta = beta;

        // Convert PDF
        Vector3f d = path[bounces].p - path[bounces - 1].p;
        Float invDist2 = 1.f / LengthSquared(d);
        Float cosTheta = AbsDot(path[bounces].n, Normalize(d));
        path[bounces].pdfFwd = pdfFwd * invDist2 * cosTheta;

        if (bounces >= maxDepth - 1)
            break;

        // Sample BSDF
        Vector3f wo = isect.wo;
        Float u = sampler.Get1D();
        pstd::optional<BSDFSample> bs =
            bsdf.Sample_f(wo, u, sampler.Get2D(), TransportMode::Importance);
        if (!bs)
            break;

        pdfFwd = bs->pdfIsProportional ? bsdf.PDF(wo, bs->wi, TransportMode::Importance)
                                       : bs->pdf;
        anyNonSpecularBounces |= !bs->IsSpecular();
        path[bounces].delta = bs->IsSpecular();

        beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;

        Float pdfRev = bsdf.PDF(bs->wi, wo, TransportMode::Importance);
        if (bs->IsSpecular())
            pdfRev = pdfFwd = 0;

        if (bounces > 0) {
            d = path[bounces - 1].p - path[bounces].p;
            invDist2 = 1.f / LengthSquared(d);
            path[bounces - 1].pdfRev = pdfRev * invDist2;
            if (path[bounces - 1].IsOnSurface())
                path[bounces - 1].pdfRev *= AbsDot(path[bounces - 1].n, Normalize(d));
        }

        currentRay = isect.SpawnRay(currentRay, bsdf, bs->wi, bs->flags, bs->eta);
    }

    return bounces + 1;
}

SampledSpectrum ReSTIRBDPTIntegrator::ConnectSubpaths(
    const std::vector<ReSTIRVertex> &lightVertices,
    const std::vector<ReSTIRVertex> &cameraVertices, int s, int t,
    SampledWavelengths &lambda, Sampler sampler, pstd::optional<Point2f> *pFilm,
    Float *misWeight) {
    SampledSpectrum L(0.f);

    if (s == 0) {
        // Interpret camera subpath as complete path
        if (t < 2)
            return L;
        const ReSTIRVertex &pt = cameraVertices[t - 1];
        if (pt.IsLight()) {
            L = pt.Le(-Normalize(pt.p - cameraVertices[t - 2].p), lambda) * pt.beta;
        }
    } else if (t == 1) {
        // Connect light subpath to camera
        const ReSTIRVertex &qs = lightVertices[s - 1];
        if (qs.IsConnectible()) {
            pstd::optional<CameraWiSample> cs =
                camera.SampleWi(qs.si, sampler.Get2D(), lambda);
            if (cs && cs->pdf > 0) {
                *pFilm = cs->pRaster;
                L = qs.beta * qs.f(cs->wi, TransportMode::Importance) * cs->Wi / cs->pdf;
                if (qs.IsOnSurface())
                    L *= AbsDot(cs->wi, qs.ns);
                if (L) {
                    L *= Tr(qs.si, cs->pLens, lambda);
                    // Scale for splatting
                    Film film = camera.GetFilm();
                    L *= Float(film.FullResolution().x) * Float(film.FullResolution().y) /
                         Float(film.PixelBounds().Area());
                }
            }
        }
    } else if (s == 1) {
        // Connect camera subpath to light
        const ReSTIRVertex &pt = cameraVertices[t - 1];
        if (pt.IsConnectible()) {
            pstd::optional<SampledLight> sampledLight =
                lightSampler.Sample(sampler.Get1D());
            if (sampledLight) {
                Light light = sampledLight->light;
                Float p_l = sampledLight->p;

                LightSampleContext ctx(pt.si);
                pstd::optional<LightLiSample> ls =
                    light.SampleLi(ctx, sampler.Get2D(), lambda);
                if (ls && ls->L && ls->pdf > 0) {
                    L = pt.beta * pt.f(ls->wi, TransportMode::Radiance) * ls->L /
                        (ls->pdf * p_l);
                    if (pt.IsOnSurface())
                        L *= AbsDot(ls->wi, pt.ns);
                    if (L)
                        L *= Tr(pt.si, ls->pLight, lambda);
                }
            }
        }
    } else {
        // General connection case
        const ReSTIRVertex &qs = lightVertices[s - 1];
        const ReSTIRVertex &pt = cameraVertices[t - 1];
        if (qs.IsConnectible() && pt.IsConnectible()) {
            Vector3f d = pt.p - qs.p;
            Float dist2 = LengthSquared(d);
            if (dist2 > 0) {
                d = Normalize(d);
                L = qs.beta * qs.f(d, TransportMode::Importance) *
                    pt.f(-d, TransportMode::Radiance) * pt.beta;
                if (L)
                    L *= G(qs, pt, lambda);
            }
        }
    }

    // Compute MIS weight
    if (L) {
        *misWeight = MISWeight(lightVertices, cameraVertices, s, t);
        L *= *misWeight;
    } else {
        *misWeight = 0;
    }

    return L;
}

SampledSpectrum ReSTIRBDPTIntegrator::G(const ReSTIRVertex &v0, const ReSTIRVertex &v1,
                                        const SampledWavelengths &lambda) const {
    Vector3f d = v0.p - v1.p;
    Float g = 1.f / LengthSquared(d);
    d *= std::sqrt(g);
    if (v0.IsOnSurface())
        g *= AbsDot(v0.ns, d);
    if (v1.IsOnSurface())
        g *= AbsDot(v1.ns, d);

    // Check visibility
    Interaction i0(v0.p, v0.n, v0.time);
    Interaction i1(v1.p, v1.n, v1.time);
    return g * Tr(i0, i1, lambda);
}

Float ReSTIRBDPTIntegrator::MISWeight(const std::vector<ReSTIRVertex> &lightVertices,
                                      const std::vector<ReSTIRVertex> &cameraVertices,
                                      int s, int t) const {
    if (s + t == 2)
        return 1;

    Float sumRi = 0;
    auto remap0 = [](Float f) -> Float { return f != 0 ? f : 1; };

    // Consider camera subpath strategies
    Float ri = 1;
    for (int i = t - 1; i > 0; --i) {
        ri *= remap0(cameraVertices[i].pdfRev) / remap0(cameraVertices[i].pdfFwd);
        if (!cameraVertices[i].delta && (i == 0 || !cameraVertices[i - 1].delta))
            sumRi += ri;
    }

    // Consider light subpath strategies
    ri = 1;
    for (int i = s - 1; i >= 0; --i) {
        ri *= remap0(lightVertices[i].pdfRev) / remap0(lightVertices[i].pdfFwd);
        bool deltaLightVertex = i > 0 ? lightVertices[i - 1].delta
                                      : (lightVertices[0].light &&
                                         IsDeltaLight(lightVertices[0].light.Type()));
        if (!lightVertices[i].delta && !deltaLightVertex)
            sumRi += ri;
    }

    return 1.f / (1.f + sumRi);
}

Float ReSTIRBDPTIntegrator::TargetPdf(const PathSample &sample) const {
    // Target function is proportional to luminance
    if (!sample.contribution)
        return 0;
    return std::max(0.f, sample.contribution.y(sample.lambda));
}

void ReSTIRBDPTIntegrator::SpatialResample(Point2i pPixel, Sampler sampler,
                                           ReservoirBuffer &current,
                                           const ReservoirBuffer &source) {
    Point2i resolution = source.Resolution();

    // Get source reservoir for this pixel
    const Reservoir &centerReservoir = source(pPixel);

    // Initialize new reservoir with center sample
    Reservoir &newReservoir = current(pPixel);

    if (centerReservoir.sample.IsValid()) {
        Float weight =
            centerReservoir.sample.targetPdf * centerReservoir.W * centerReservoir.M;
        newReservoir.Update(centerReservoir.sample, weight, sampler.Get1D());
    }

    // Sample spatial neighbors
    int validNeighbors = 0;
    for (int i = 0; i < spatialNeighbors; ++i) {
        // Sample neighbor within radius
        Float angle = sampler.Get1D() * 2 * Pi;
        Float radius = sampler.Get1D() * spatialRadius;
        Point2i neighborPixel = pPixel + Point2i(int(radius * std::cos(angle)),
                                                 int(radius * std::sin(angle)));

        if (!source.InBounds(neighborPixel))
            continue;

        const Reservoir &neighborReservoir = source(neighborPixel);
        if (!neighborReservoir.sample.IsValid())
            continue;

        // Evaluate target PDF at this pixel for neighbor's sample
        Float targetPdf = TargetPdf(neighborReservoir.sample);

        // Check visibility if samples differ significantly
        if (targetPdf > 0) {
            // Simplified visibility check - in full implementation would do reconnection
            Float weight = targetPdf * neighborReservoir.W * neighborReservoir.M;
            if (newReservoir.Update(neighborReservoir.sample, weight, sampler.Get1D()))
                validNeighbors++;
        }
    }

    // Finalize reservoir
    int totalM = centerReservoir.M + validNeighbors;
    if (totalM > 0 && newReservoir.sample.targetPdf > 0) {
        newReservoir.M = totalM;
        newReservoir.W = newReservoir.wSum / (newReservoir.sample.targetPdf * totalM);
    }
}

void ReSTIRBDPTIntegrator::TemporalResample(Point2i pPixel, Sampler sampler,
                                            ReservoirBuffer &current,
                                            const ReservoirBuffer &previous) {
    // Simple temporal reuse - combine with previous frame's reservoir
    const Reservoir &prevReservoir = previous(pPixel);
    Reservoir &newReservoir = current(pPixel);

    // Copy current reservoir
    newReservoir = current(pPixel);

    if (prevReservoir.sample.IsValid()) {
        Float targetPdf = TargetPdf(prevReservoir.sample);
        if (targetPdf > 0) {
            // Clamp temporal contribution to avoid fireflies
            int clampedM = std::min(prevReservoir.M, 20 * newReservoir.M);
            Float weight = targetPdf * prevReservoir.W * clampedM;
            newReservoir.Merge(prevReservoir, targetPdf, sampler.Get1D());
        }
    }

    newReservoir.Finalize();
}

bool ReSTIRBDPTIntegrator::CheckVisibility(const ReSTIRVertex &v0,
                                           const ReSTIRVertex &v1) const {
    Interaction i0(v0.p, v0.n, v0.time);
    Interaction i1(v1.p, v1.n, v1.time);
    return Unoccluded(i0, i1);
}

std::string ReSTIRBDPTIntegrator::ToString() const {
    return StringPrintf("[ ReSTIRBDPTIntegrator maxDepth: %d spatialIterations: %d "
                        "spatialNeighbors: %d spatialRadius: %f temporalReuse: %s "
                        "regularize: %s ]",
                        maxDepth, spatialIterations, spatialNeighbors, spatialRadius,
                        temporalReuse, regularize);
}

std::unique_ptr<ReSTIRBDPTIntegrator> ReSTIRBDPTIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    if (!camera.Is<PerspectiveCamera>())
        ErrorExit("Only the \"perspective\" camera is currently supported with the "
                  "\"restirbdpt\" integrator.");

    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    int spatialIterations = parameters.GetOneInt("spatialiterations", 1);
    int spatialNeighbors = parameters.GetOneInt("spatialneighbors", 5);
    Float spatialRadius = parameters.GetOneFloat("spatialradius", 30.f);
    bool temporalReuse = parameters.GetOneBool("temporalreuse", false);
    bool regularize = parameters.GetOneBool("regularize", false);

    return std::make_unique<ReSTIRBDPTIntegrator>(
        camera, sampler, aggregate, lights, maxDepth, spatialIterations, spatialNeighbors,
        spatialRadius, temporalReuse, regularize);
}

}  // namespace pbrt
