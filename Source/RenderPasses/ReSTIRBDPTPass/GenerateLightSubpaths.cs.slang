import PathReservoir;
import Scene.Scene;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.EmissivePowerSampler;
import Utils.Math.MathHelpers;
import Utils.Math.HashUtils;
import Scene.RaytracingInline;
import LRM;

#include "Utils/Math/MathConstants.slangh"

cbuffer CB
{
    int gNumLightSubpaths;
    int gNumBounces;
    float gMISPower;
    int gFrameCount;
    uint2 gTargetDim;
    float gRoughnessThreshold;
}

AppendStructuredBuffer<LightSubpathVertex> gLVC;

ParameterBlock<LRMData> gLRM;

ParameterBlock<EmissivePowerSampler> gEmissivePowerSampler;

RWStructuredBuffer<uint> gDebugCounter;

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= (uint)gNumLightSubpaths)
        return;

    SampleGenerator sg = SampleGenerator(uint2(i, 0), gFrameCount);
    LightSubpathVertex yj;

    if (!InitializeLightPath(sg, i, yj))
        return;

    for (int j = 1; j < gNumBounces; ++j)
    {
        bool rough = false;

        if (!TraceNewVertex(sg, yj, rough))
            break;

        if (rough)
        {
            gLVC.Append(yj);
            PathReservoir r;
            uint k;
            if (!ConnectToCamera(yj, sg, r, k))
                continue;

            gLRM.Insert(k, r);
        }
    }
}

bool InitializeLightPath(inout SampleGenerator sg, uint subpathId, out LightSubpathVertex y)
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

    SceneVertex y0 = {};
    y0.bary = barycentrics;
    y0.triangleIndex = triangleIndex;
    y.y0 = y0;

    float3 posW = tri.getPosition(barycentrics);
    float3 normal = tri.normal;
    float3 faceNormal = tri.normal;

    y.subpathPdf = triangleSelectionPdf * (1.0f / tri.area);
    y.subpathSeed = jenkinsHash(subpathId ^ jenkinsHash(gFrameCount));
    y.subpathId = subpathId;

    y.diffuseBounces = 1;

    y.numVertices = 1;
    y.numBounces = 0;

    y.dVC = 1;
    y.dp = 0;

    return true;
}

bool TraceNewVertex(inout SampleGenerator sg, inout LightSubpathVertex yj, out bool rough)
{
    LightSubpathVertex y_j1 = yj;

    if (yj.numBounces == 0)
    {
        EmissiveTriangle tri = gScene.lightCollection.getTriangle(yj.y0.triangleIndex);
        float3 lightPos = tri.getPosition(yj.y0.bary);
        float3 lightNormal = tri.normal;

        float2 uDir = sampleNext2D(sg);
        float pdfSA;
        float3 localDir = sample_cosine_hemisphere_concentric(uDir, pdfSA);

        ShadingFrame sf;
        sf.N = normalize(lightNormal);
        sf.T = perp_stark(sf.N);
        sf.B = cross(sf.N, sf.T);
        float3 worldDir = sf.fromLocal(localDir);

        SceneRayQuery<0> rq;
        Ray ray = Ray(lightPos, worldDir, 1e-4f, 10000.0f);

        float hitT;
        HitInfo hit = rq.traceRay(ray, hitT);

        if (!hit.isValid())
            return false;

        TriangleHit triHit = hit.getTriangleHit();
        VertexData vd = gScene.getVertexData(triHit);
        uint hitMaterialID = gScene.getMaterialID(triHit.instanceID);
        ShadingData sdHit = gScene.materials.prepareShadingData(vd, hitMaterialID, -worldDir);

        let miHit = gScene.materials.getMaterialInstance(sdHit, ExplicitLodTextureSampler(0.f));
        BSDFProperties hitProps = miHit.getProperties(sdHit);
        float rgh = hitProps.roughness;

        rough = (rgh > gRoughnessThreshold);

        if (rough)
        { 
            yj.diffuseBounces |= (1u << (yj.numBounces + 1));
        }

        PathVertex y_s1;
        y_s1.instanceID = triHit.instanceID;
        y_s1.materialID = gScene.getMaterialID(triHit.instanceID);
        y_s1.posW = vd.posW;
        y_s1.primitiveID = triHit.primitiveIndex;
        y_s1.dirIn = -ray.dir;
        y_s1.barycentrics = triHit.barycentrics;

        yj.y_s1 = y_s1;

        float2 uv = tri.getTexCoord(yj.y0.bary);
        float3 Le = gScene.materials.evalEmissive(tri.materialID, uv);

        yj.throughput = Le * (M_PI / yj.subpathPdf);

        float cosThetaIn = abs(dot(vd.normalW, -worldDir));
        float distSq = max(hitT * hitT, 1e-7f);
        float pdfFwdArea = pdfSA * (cosThetaIn / distSq);

        float p0_trace = yj.subpathPdf * pdfSA;
        float p0_connect = 1.0f;

        yj.dp = p0_connect / max(p0_trace * pdfFwdArea, 1e-10f);
        yj.dVC = (cosThetaIn / distSq) / max(p0_trace * pdfFwdArea, 1e-10f);

        yj.subpathPdf *= pdfFwdArea;
    }
    else
    {
        TriangleHit currentHit;
        currentHit.instanceID = yj.y_s1.instanceID;
        currentHit.primitiveIndex = yj.y_s1.primitiveID;
        currentHit.barycentrics = yj.y_s1.barycentrics;

        VertexData vCurrent = gScene.getVertexData(currentHit);
        ShadingData sdCurrent = gScene.materials.prepareShadingData(vCurrent, yj.y_s1.materialID, yj.y_s1.dirIn);
        let mi = gScene.materials.getMaterialInstance(sdCurrent, ExplicitLodTextureSampler(0.f));

        BSDFSample bsdfSample;
        if (!mi.sample(sdCurrent, sg, bsdfSample) || bsdfSample.pdf == 0.0f)
            return false;

        SceneRayQuery<0> rq;
        Ray ray = Ray(sdCurrent.posW, bsdfSample.wo, 1e-4f, 10000.0f);
        float hitT;
        HitInfo hit = rq.traceRay(ray, hitT);

        if (!hit.isValid())
            return false;

        TriangleHit triHitNext = hit.getTriangleHit();
        VertexData vNext = gScene.getVertexData(triHitNext);
        uint nextMaterialID = gScene.getMaterialID(triHitNext.instanceID);
        ShadingData sdNext = gScene.materials.prepareShadingData(vNext, nextMaterialID, -bsdfSample.wo);

        float cosThetaOut = abs(dot(sdCurrent.frame.N, bsdfSample.wo));
        float cosThetaIn = abs(dot(sdNext.frame.N, -bsdfSample.wo));
        float distSq = max(hitT * hitT, 1e-7f);

        float G = (cosThetaOut * cosThetaIn) / distSq;
        float pdfFwdArea = bsdfSample.pdf * (cosThetaIn / distSq);

        let miNext = gScene.materials.getMaterialInstance(sdNext, ExplicitLodTextureSampler(0.f));
        float pdfRevSA = miNext.evalPdf(sdNext, -bsdfSample.wo);

        bool isNonDelta = !bsdfSample.isLobe(LobeType::Delta);
        float nonDelta = isNonDelta ? 1.0f : 0.0f;
        float invPdfFwdArea = 1.0f / max(pdfFwdArea, 1e-10f);

        float ratio = cosThetaOut / max(bsdfSample.pdf, 1e-10f);

        float beta = float(gMISPower);
        float dVC_new = pow(ratio, beta) * (nonDelta * yj.dp + pow(pdfRevSA, beta) * yj.dVC);

        float dp_new = nonDelta * pow(invPdfFwdArea, beta);

        yj.dVC = dVC_new;
        yj.dp = dp_new;

        BSDFProperties nextProps = miNext.getProperties(sdNext);
        float rgh = nextProps.roughness;
        float p_terminate = clamp(0.2f * rgh, 0.0f, 0.95f);
        float p_survive = 1.0f - p_terminate;

        if (sampleNext1D(sg) < p_terminate)
            return false;

        rough = (rgh > 0.08f);

        yj.throughput = (yj.throughput * bsdfSample.weight) / p_survive;
        yj.subpathPdf *= (pdfFwdArea * p_survive);

        yj.y_s1.instanceID = triHitNext.instanceID;
        yj.y_s1.materialID = nextMaterialID;
        yj.y_s1.posW = sdNext.posW;
        yj.y_s1.primitiveID = triHitNext.primitiveIndex;
        yj.y_s1.dirIn = -ray.dir;
        yj.y_s1.barycentrics = triHitNext.barycentrics;
    }

    yj.numBounces++;
    yj.numVertices++;
    return true;
}

bool ConnectToCamera(in LightSubpathVertex yj, inout SampleGenerator sg, out PathReservoir r, out uint k)
{
    float3 cameraPos = gScene.camera.data.posW;
    float3 dirToCamera = cameraPos - yj.y_s1.posW;

    float distSq = max(dot(dirToCamera, dirToCamera), 1e-7f);
    float dist = sqrt(distSq);
    dirToCamera /= dist;

    float4 clipPos = mul(float4(yj.y_s1.posW, 1.0f), gScene.camera.data.viewProjMatNoJitter);

    if (clipPos.w <= 0.0f)
        return false;

    float2 ndc = clipPos.xy / clipPos.w;

    if (ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f)
        return false;

    float2 uv = ndc * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;

    uint2 pixelCoord = uint2(uv * float2(gTargetDim.x, gTargetDim.y));

    if (pixelCoord.x >= gTargetDim.x || pixelCoord.y >= gTargetDim.y)
        return false;

    k = pixelCoord.y * gTargetDim.x + pixelCoord.x;

    TriangleHit currentHit;
    currentHit.instanceID = yj.y_s1.instanceID;
    currentHit.primitiveIndex = yj.y_s1.primitiveID;
    currentHit.barycentrics = yj.y_s1.barycentrics;
    VertexData vCurrent = gScene.getVertexData(currentHit);

    ShadingData sdCurrent = gScene.materials.prepareShadingData(vCurrent, yj.y_s1.materialID, yj.y_s1.dirIn);
    let mi = gScene.materials.getMaterialInstance(sdCurrent, ExplicitLodTextureSampler(0.f));

    float3 bsdfWeight = mi.eval(sdCurrent, dirToCamera, sg);
    if (all(bsdfWeight == 0.0f))
        return false;

    float pdfRevSA = mi.evalPdf(sdCurrent, dirToCamera);

    SceneRayQuery<0> rq;
    Ray ray = Ray(yj.y_s1.posW, dirToCamera, 1e-4f, dist - 1e-4f);

    if (!rq.traceVisibilityRay(ray))
        return false;

    float cosThetaCam = dot(gScene.camera.data.cameraW, -dirToCamera);
    if (cosThetaCam <= 0.0f)
        return false;

    float cosThetaCam4 = cosThetaCam * cosThetaCam * cosThetaCam * cosThetaCam;
    float cameraImportance = 1.0f / max(cosThetaCam4, 1e-6f);

    float3 unweightedContribution = yj.throughput * bsdfWeight * cameraImportance;
    if (all(unweightedContribution == 0.0f))
        return false;

    float cosThetaOut = abs(dot(sdCurrent.frame.N, dirToCamera));

    float p_revArea = pdfRevSA * (cosThetaCam / distSq);
    float w_light = (p_revArea / max(float(gNumLightSubpaths), 1.0f)) * (yj.dp + pdfRevSA * yj.dVC);
    float misWeight = 1.0f / (w_light + 1.0f);

    float invPdf = 1.0f / max(float(gNumLightSubpaths) * yj.subpathPdf, 1e-15f);

    bool isCaustic = false;
    if (yj.numBounces > 0)
    { 
        bool isPrevRough = (yj.diffuseBounces & (1u << (yj.numBounces - 1))) != 0;
        isCaustic = !isPrevRough;
    }

    InitializeReservoir(isCaustic, r);

    r.UCW = invPdf;

    r.confidenceWeight = 1.0f;

    r.pathF = unweightedContribution;
    r.misWeight = misWeight;

    r.bounces = yj.numBounces + 1;

    r.prefixBounces = 0;

    r.prefixDiffuseBounces = 0;

    r.xr = yj.y_s1;
    r.y0 = yj.y0;
    r.dirOut = dirToCamera;
    r.reconnectionCos = cosThetaOut;
    r.reconnectionDist = dist;

    r.lightPathSeed = yj.subpathSeed;
    r.lightPathId = yj.subpathId;
    r.lightPathPdf = yj.subpathPdf;
    r.lightPathVertices = yj.numVertices;
    r.lightPathDiffuseBounces = r.lightPathDiffuseBounces = countbits(yj.diffuseBounces);
    ;

    r.cameraSubpathSeed = 0;
    r.cameraSubpathId = 0;
    r.prefixPdf = 1.0f;
    r.suffixF = float3(0.0f);
    r.suffixPdf = 0.0f;

    MisEvalData misData = {};

    misData.d_p_light = yj.dp;
    misData.d_vc_light = yj.dVC;
    misData.p_rev_light = p_revArea;

    misData.p_rev_sigma_light = pdfRevSA;

    misData.p0_trace_cam = 1.0f;
    misData.p0_connect_cam = 1.0f;

    r.cachedMisData = misData;

    return true;
}
