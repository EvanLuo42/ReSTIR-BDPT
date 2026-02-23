import PathReservoir;
import LRM;
import Utils.Sampling.SampleGenerator;
import Scene.Scene;
import Scene.RaytracingInline;
import Rendering.Lights.EmissivePowerSampler;
import Utils.Math.MathHelpers;

#include "Utils/Math/MathConstants.slangh"

cbuffer CB
{
    int gNumBounces;
    uint gNumLightSubpaths;
    uint2 gTargetDim;
    uint gFrameCount;
    float gRoughnessThreshold;
    float gMISPower;
}

StructuredBuffer<LightSubpathVertex> gLVC;

RWStructuredBuffer<PathReservoir> gOutputReservoirs;
RWStructuredBuffer<PathReservoir> gOutputCausticReservoirs;

ParameterBlock<LRMData> gLRM;
ParameterBlock<EmissivePowerSampler> gEmissivePowerSampler;

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixelCoord = tid.xy;

    if (pixelCoord.x >= gTargetDim.x || pixelCoord.y >= gTargetDim.y)
        return;

    uint i = pixelCoord.y * gTargetDim.x + pixelCoord.x;
    SampleGenerator sg = SampleGenerator(uint2(i, 0), gFrameCount);

    PathReservoir ri;

    InitializeReservoir(false, ri);

    CameraSubpathVertex xj;

    if (!InitializeCameraPath(xj))
        return;

    for (int i = 1; i < gNumBounces; ++i)
    {
        bool rough = false;
        bool emissive = false;
        bool terminate = false;

        if (!TraceNewVertex(sg, xj, pixelCoord, rough, emissive, terminate))
            break;

        if (rough)
        {
            PathReservoir r_NEE;
            if (ConnectToRandomLight(xj, sg, r_NEE))
                ri.Merge(sg, r_NEE);

            PathReservoir r_LVC;
            if (ConnectToRandomLVC(xj, sg, r_LVC))
                ri.Merge(sg, r_LVC);
        }

        if (emissive)
        {
            PathReservoir r;
            EvalEmission(xj, r);
            ri.Merge(sg, r);
        }

        if (terminate)
            break;
    }

    PathReservoir ci;

    InitializeReservoir(true, ci);

    uint numLightPaths = gLRM.GetReservoirCount(i);

    for (uint j = 0; j < numLightPaths; ++j)
    {
        PathReservoir r_LT = gLRM.GetReservoir(i, j);

        r_LT.UCW *= 1.0f / max(float(gNumLightSubpaths), 1.0f);

        if (r_LT.flags.isCaustic())
        {
            ci.Merge(sg, r_LT);
        }
        else
        {
            ri.Merge(sg, r_LT);
        }
    }

    ri.FinalizeResampling();
    ci.FinalizeResampling();

    ri.confidenceWeight = 1.0f;
    ci.confidenceWeight = 1.0f;

    gOutputReservoirs[i] = ri;
    gOutputCausticReservoirs[i] = ci;
}

bool InitializeCameraPath(out CameraSubpathVertex x0)
{
    x0.throughput = float3(1.0f, 1.0f, 1.0f);

    x0.numVertices = 1;
    x0.numBounces = 0;

    x0.subpathPdf = 1.0f;

    x0.dVC = 0.0f;
    x0.dp = 0.0f;

    x0.gamma = 0.0f;
    x0.lambda_vc = 0.0f;
    x0.lambda_p = 0.0f;
    x0.sigma = 0.0f;

    return true;
}

bool TraceNewVertex(
    inout SampleGenerator sg,
    inout CameraSubpathVertex xj,
    uint2 pixel,
    out bool rough,
    out bool emissive,
    out bool terminate
)
{
    Ray ray;
    float pdfSA = 1.0f;
    float3 bsdfWeight = float3(1.0f);
    bool isNonDelta = true;
    float pdfRevSA = 0.0f;
    float cosThetaOut = 1.0f;
    VertexData vCurrent;

    if (xj.numVertices == 1)
    {
        ray = gScene.camera.computeRayPinhole(pixel, gTargetDim, true);

        float cosThetaCam = dot(normalize(gScene.camera.data.cameraW), ray.dir);
        if (abs(cosThetaCam) <= 0.0f)
            return false;

        float cosThetaCam4 = cosThetaCam * cosThetaCam * cosThetaCam * cosThetaCam;
        bsdfWeight = float3(1.0f / max(cosThetaCam4, 1e-6f));
    }
    else
    {
        TriangleHit currentHit;
        currentHit.instanceID = xj.x.instanceID;
        currentHit.primitiveIndex = xj.x.primitiveID;
        currentHit.barycentrics = xj.x.barycentrics;

        vCurrent = gScene.getVertexData(currentHit);
        ShadingData sdCurrent = gScene.materials.prepareShadingData(vCurrent, xj.x.materialID, xj.x.dirIn);
        let mi = gScene.materials.getMaterialInstance(sdCurrent, ExplicitLodTextureSampler(0.f));

        BSDFSample bsdfSample;
        if (!mi.sample(sdCurrent, sg, bsdfSample))
            return false;

        isNonDelta = bsdfSample.isLobe(LobeType::NonDelta);

        ray = Ray(sdCurrent.posW, bsdfSample.wo);
        pdfSA = bsdfSample.pdf;
        bsdfWeight = bsdfSample.weight;
        isNonDelta = !bsdfSample.isLobe(LobeType::Delta);

        cosThetaOut = abs(dot(sdCurrent.frame.N, bsdfSample.wo));
    }

    SceneRayQuery<0> rq;
    float hitT;
    HitInfo hit = rq.traceRay(ray, hitT);

    if (!hit.isValid())
        return false;

    TriangleHit triHitNext = hit.getTriangleHit();
    VertexData vNext = gScene.getVertexData(triHitNext);
    uint nextMaterialID = gScene.getMaterialID(triHitNext.instanceID);
    ShadingData sdNext = gScene.materials.prepareShadingData(vNext, nextMaterialID, -ray.dir);
    let miNext = gScene.materials.getMaterialInstance(sdNext, ExplicitLodTextureSampler(0.f));

    BSDFProperties nextProps = miNext.getProperties(sdNext);
    float rgh = nextProps.roughness;

    rough = rgh > gRoughnessThreshold;

    float2 uv = vNext.texC;
    float3 Le = gScene.materials.evalEmissive(nextMaterialID, uv);
    emissive = any(Le > 0.0f);

    float cosThetaIn = abs(dot(sdNext.frame.N, -ray.dir));
    float distSq = max(hitT * hitT, 1e-4f);

    float G = (cosThetaOut * cosThetaIn) / distSq;
    float pdfFwdArea = pdfSA * (cosThetaIn / distSq);
    pdfRevSA = miNext.evalPdf(sdNext, -ray.dir);

    float p_survive = 1.0f;
    if (xj.numVertices > 1)
    {
        float p_terminate = clamp(0.2f * rgh, 0.0f, 0.95f);
        p_survive = 1.0f - p_terminate;
        terminate = sampleNext1D(sg) < p_terminate;
    }

    float nonDelta = isNonDelta ? 1.0f : 0.0f;
    float invPdfFwdArea = 1.0f / max(pdfFwdArea, 1e-10f);

    if (xj.numVertices == 1)
    {
        float invPdfFwdArea = 1.0f / max(pdfFwdArea, 1e-10f);

        xj.dp = pow(float(gNumLightSubpaths) / pdfFwdArea, gMISPower);
        xj.dVC = 0.0f;

        xj.gamma = 0.0f;
        xj.lambda_p = 0.0f;
        xj.lambda_vc = 0.0f;
        xj.sigma = 0.0f;
    }
    else
    {
        float invPdfFwdArea = 1.0f / max(pdfFwdArea, 1e-10f);

        float invPdfFwdAreaPow = pow(invPdfFwdArea, gMISPower);

        float pdfRevPow = pow(max(pdfRevSA, 0.0f), gMISPower);

        float gRev = cosThetaOut / distSq;

        float gRevPow = pow(max(gRev, 0.0f) / max(pdfFwdArea, 1e-10f), gMISPower);

        float old_dp = xj.dp;
        float old_dVC = xj.dVC;

        if (isNonDelta)
        {
            xj.dp = invPdfFwdAreaPow;
            xj.dVC = gRevPow * (old_dp + pdfRevPow * old_dVC);
        }
        else
        {
            xj.dp = 0.0f;
            xj.dVC = old_dVC;
        }
    }

    xj.throughput = (xj.throughput * bsdfWeight) / p_survive;

    xj.numVertices++;
    xj.numBounces += (uint16_t)(rough ? 1 : 0);

    xj.x.instanceID = triHitNext.instanceID;
    xj.x.materialID = nextMaterialID;
    xj.x.posW = sdNext.posW;
    xj.x.primitiveID = triHitNext.primitiveIndex;
    xj.x.dirIn = -ray.dir;
    xj.x.barycentrics = triHitNext.barycentrics;

    return true;
}

bool ConnectToRandomLight(in CameraSubpathVertex xj, inout SampleGenerator sg, out PathReservoir r)
{
    if (gScene.lightCollection.isEmpty())
        return false;

    EmissivePower emissivePower = gEmissivePowerSampler._emissivePower;

    float uLight = sampleNext1D(sg);
    uint triangleCount = gScene.lightCollection.triangleCount;
    uint triangleIndex = min((uint)(uLight * triangleCount), triangleCount - 1);

    uint2 packed = emissivePower.triangleAliasTable[triangleIndex];
    float threshold = f16tof32(packed.x >> 16u);
    uint selectAbove = ((packed.x & 0xFFFFu) << 8u) | ((packed.y >> 24u) & 0xFFu);
    uint selectBelow = packed.y & 0xFFFFFFu;

    triangleIndex = (sampleNext1D(sg) >= threshold) ? selectAbove : selectBelow;

    float triangleSelectionPdf = gScene.lightCollection.fluxData[triangleIndex].flux * emissivePower.invWeightsSum;

    EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);
    float2 uPos = sampleNext2D(sg);
    float3 barycentrics = sample_triangle(uPos);

    float3 lightPos = tri.getPosition(barycentrics);
    float3 lightNormal = tri.normal;

    float pdfLightArea = triangleSelectionPdf * (1.0f / tri.area);

    float3 dirToLight = lightPos - xj.x.posW;
    float distSq = max(dot(dirToLight, dirToLight), 1e-7f);
    float dist = sqrt(distSq);
    dirToLight /= dist;

    float cosThetaIn = dot(lightNormal, -dirToLight);
    if (cosThetaIn <= 0.0f)
        return false;

    float3 Tr = EvalTransmittance(xj.x.posW, dirToLight, dist - 1e-4f);
    if (all(Tr == 0.0f))
        return false;

    TriangleHit currentHit;
    currentHit.instanceID = xj.x.instanceID;
    currentHit.primitiveIndex = xj.x.primitiveID;
    currentHit.barycentrics = xj.x.barycentrics;

    VertexData vCurrent = gScene.getVertexData(currentHit);

    ShadingData sdCurrent = gScene.materials.prepareShadingData(vCurrent, xj.x.materialID, xj.x.dirIn);
    let mi = gScene.materials.getMaterialInstance(sdCurrent, ExplicitLodTextureSampler(0.f));

    float3 bsdfWeight = mi.eval(sdCurrent, dirToLight, sg);
    if (all(bsdfWeight == 0.0f))
        return false;

    ShadingData sdRev = gScene.materials.prepareShadingData(vCurrent, xj.x.materialID, dirToLight);
    let miRev = gScene.materials.getMaterialInstance(sdRev, ExplicitLodTextureSampler(0.f));
    float pdfRevSA = miRev.evalPdf(sdRev, xj.x.dirIn);

    float2 uv = tri.getTexCoord(barycentrics);
    float3 Le = gScene.materials.evalEmissive(tri.materialID, uv);

    InitializeReservoir(false, r);

    r.UCW = 1.0f / max(pdfLightArea, 1e-10f);
    r.confidenceWeight = 1.0f;

    float G = cosThetaIn / distSq;
    float3 pathF = xj.throughput * bsdfWeight * G * Le * Tr;
    r.pathF = pathF;

    float pathRatio = float(gNumLightSubpaths) / float(gTargetDim.x * gTargetDim.y);
    float prefix = pow(pdfLightArea * pathRatio, gMISPower);
    float w_light = prefix * (xj.dp + pow(pdfRevSA, gMISPower) * xj.dVC);
    float misWeight = 1.0f / (1.0f + w_light);

    r.misWeight = misWeight;

    // $s=1$ (Light subpath) + $t$ (Camera subpath)
    r.bounces = xj.numVertices - 1;
    r.prefixBounces = xj.numVertices - 1;
    r.prefixDiffuseBounces = xj.numBounces;

    r.dirOut = 0;
    r.reconnectionCos = 0;
    r.reconnectionDist = 0;

    SceneVertex y0 = {};
    y0.bary = barycentrics;
    y0.triangleIndex = triangleIndex;
    r.y0 = y0;

    r.lightPathVertices = 1;

    r.lightPathDiffuseBounces = 0;
    r.lightPathPdf = pdfLightArea;
    r.lightPathSeed = 0;

    r.lightPathId = 0;

    r.prefixPdf = xj.subpathPdf;

    MisEvalData misData = {};

    misData.d_p_cam = xj.dp;
    misData.d_vc_cam = xj.dVC;
    misData.p_fwd_cam = 0;
    misData.p_rev_sigma_cam = pdfRevSA;
    misData.gamma = xj.gamma;
    misData.lambda_vc = xj.lambda_vc;
    misData.lambda_p = xj.lambda_p;
    misData.sigma = xj.sigma;

    misData.p_rev_light = pdfLightArea;
    misData.p0_connect_light = 1.0f;
    misData.p0_trace_light = 1.0f;

    r.cachedMisData = misData;

    return true;
}

bool ConnectToRandomLVC(in CameraSubpathVertex xj, inout SampleGenerator sg, out PathReservoir r)
{
    uint lvcSize, stride;
    gLVC.GetDimensions(lvcSize, stride);

    if (lvcSize == 0)
        return false;

    uint lvcIndex = min((uint)(sampleNext1D(sg) * lvcSize), lvcSize - 1);
    LightSubpathVertex yk = gLVC[lvcIndex];

    float3 dirToLight = yk.y_s1.posW - xj.x.posW;
    float distSq = max(dot(dirToLight, dirToLight), 1e-7f);
    float dist = sqrt(distSq);
    dirToLight /= dist;

    float3 Tr = EvalTransmittance(xj.x.posW, dirToLight, dist - 1e-4f);
    if (all(Tr == 0.0f))
        return false;

    TriangleHit camHit;
    camHit.instanceID = xj.x.instanceID;
    camHit.primitiveIndex = xj.x.primitiveID;
    camHit.barycentrics = xj.x.barycentrics;
    VertexData vCam = gScene.getVertexData(camHit);

    ShadingData sdCam = gScene.materials.prepareShadingData(vCam, xj.x.materialID, xj.x.dirIn);
    let miCam = gScene.materials.getMaterialInstance(sdCam, ExplicitLodTextureSampler(0.f));

    float3 f_cam = miCam.eval(sdCam, dirToLight, sg);

    if (all(f_cam == 0.0f))
        return false;
    float pdfFwdSA_cam = miCam.evalPdf(sdCam, dirToLight);

    ShadingData sdCamRev = gScene.materials.prepareShadingData(vCam, xj.x.materialID, dirToLight);
    let miCamRev = gScene.materials.getMaterialInstance(sdCamRev, ExplicitLodTextureSampler(0.f));
    float pdfRevSA_cam = miCamRev.evalPdf(sdCamRev, xj.x.dirIn);

    TriangleHit lightHit;
    lightHit.instanceID = yk.y_s1.instanceID;
    lightHit.primitiveIndex = yk.y_s1.primitiveID;
    lightHit.barycentrics = yk.y_s1.barycentrics;
    VertexData vLight = gScene.getVertexData(lightHit);

    ShadingData sdLight = gScene.materials.prepareShadingData(vLight, yk.y_s1.materialID, yk.y_s1.dirIn);
    let miLight = gScene.materials.getMaterialInstance(sdLight, ExplicitLodTextureSampler(0.f));

    float3 f_light = miLight.eval(sdLight, -dirToLight, sg);
    if (all(f_light == 0.0f))
        return false;

    InitializeReservoir(false, r);

    r.confidenceWeight = 1.0f;

    float cosThetaCam = abs(dot(sdCam.frame.N, dirToLight));

    float cosThetaLight = abs(dot(sdLight.frame.N, -dirToLight));

    float G = 1.0f / distSq;

    float3 pathF = xj.throughput * f_cam * G * f_light * yk.throughput * Tr;
    r.pathF = pathF;

    r.UCW = (float)lvcSize / max((float)gNumLightSubpaths, 1.0f);

    float pdfRevSA_light = miLight.evalPdf(sdLight, -dirToLight);
    float p_rev_light = pdfRevSA_light * (cosThetaCam / distSq);
    float p_fwd_cam = pdfFwdSA_cam * (cosThetaLight / distSq);
    float w_cam = pow(p_rev_light, gMISPower) * (yk.dp + pow(pdfRevSA_light, gMISPower) * yk.dVC);
    float w_light = pow(p_fwd_cam, gMISPower) * (xj.dp + pow(pdfRevSA_cam, gMISPower) * xj.dVC);
    float misWeight = 1.0f / (1.0f + w_cam + w_light);

    r.misWeight = misWeight;

    r.bounces = xj.numVertices + yk.numVertices - 2;
    r.prefixBounces = xj.numVertices - 1;
    r.prefixDiffuseBounces = xj.numBounces;

    r.xr = yk.y_s1;
    r.dirOut = dirToLight;
    r.reconnectionCos = abs(dot(sdLight.frame.N, -dirToLight));
    r.reconnectionDist = dist;

    r.y0 = yk.y0;
    r.lightPathSeed = yk.subpathSeed;
    r.lightPathId = yk.subpathId;
    r.lightPathPdf = yk.subpathPdf;
    r.lightPathVertices = yk.numVertices;
    r.lightPathDiffuseBounces = yk.numBounces;

    r.prefixPdf = xj.subpathPdf;

    MisEvalData misData = {};

    misData.p_fwd_cam = p_fwd_cam;
    misData.d_p_cam = xj.dp;
    misData.d_vc_cam = xj.dVC;
    misData.p_rev_sigma_cam = pdfRevSA_cam;
    misData.gamma = xj.gamma;
    misData.lambda_vc = xj.lambda_vc;
    misData.lambda_p = xj.lambda_p;
    misData.sigma = xj.sigma;

    misData.d_p_light = yk.dp;
    misData.d_vc_light = yk.dVC;

    misData.p_rev_light = p_rev_light;
    misData.p_rev_sigma_light = pdfRevSA_light;

    misData.p0_trace_cam = 1.0f;
    misData.p0_connect_cam = 1.0f;
    misData.p0_trace_light = 1.0f;
    misData.p0_connect_light = 1.0f;

    r.cachedMisData = misData;

    return true;
}

bool EvalEmission(in CameraSubpathVertex xj, out PathReservoir r)
{
    TriangleHit hit;
    hit.instanceID = xj.x.instanceID;
    hit.primitiveIndex = xj.x.primitiveID;
    hit.barycentrics = xj.x.barycentrics;

    VertexData vd = gScene.getVertexData(hit);
    float3 Le = gScene.materials.evalEmissive(xj.x.materialID, vd.texC);

    float pdfLightArea = 0.0f;
    uint triangleIndex = gScene.lightCollection.getTriangleIndex(xj.x.instanceID, xj.x.primitiveID);

    if (triangleIndex != MeshLightData::kInvalidIndex)
    {
        EmissivePower emissivePower = gEmissivePowerSampler._emissivePower;

        float triangleSelectionPdf = gScene.lightCollection.fluxData[triangleIndex].flux * emissivePower.invWeightsSum;
        EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);

        pdfLightArea = triangleSelectionPdf * (1.0f / tri.area);
    }

    InitializeReservoir(false, r);

    r.confidenceWeight = 1.0f;

    r.pathF = xj.throughput * Le;
    r.UCW = 1.0f;

    float w_light = pow(pdfLightArea, gMISPower) * xj.dp;

    r.misWeight = 1.0f / (1.0f + w_light);

    r.bounces = xj.numVertices - 1;
    r.prefixBounces = xj.numVertices - 1;
    r.prefixDiffuseBounces = xj.numBounces;

    r.xr = xj.x;

    SceneVertex y0 = {};
    float2 bary = xj.x.barycentrics;
    y0.bary = float3(1.0f - bary.x - bary.y, bary.x, bary.y);
    y0.triangleIndex = triangleIndex;
    r.y0 = y0;

    r.lightPathVertices = 0;
    r.lightPathDiffuseBounces = 0;
    r.lightPathPdf = 0.0f;
    r.lightPathSeed = 0;
    r.lightPathId = 0;

    r.prefixPdf = xj.subpathPdf;
    r.dirOut = float3(0.0f);
    r.reconnectionCos = 0.0f;
    r.reconnectionDist = 0.0f;

    MisEvalData misData = {};

    misData.d_p_cam = xj.dp;
    misData.d_vc_cam = xj.dVC;
    misData.p_fwd_cam = 0.0f;
    misData.p_rev_sigma_cam = 0.0f;

    misData.gamma = xj.gamma;
    misData.lambda_vc = xj.lambda_vc;
    misData.lambda_p = xj.lambda_p;
    misData.sigma = xj.sigma;

    misData.p_rev_light = pdfLightArea;
    misData.p0_connect_light = 1.0f;
    misData.p0_trace_light = 1.0f;
    misData.p0_trace_cam = 1.0f;
    misData.p0_connect_cam = 1.0f;

    r.cachedMisData = misData;

    return true;
}
