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

    y.numVertices = 1;
    y.numBounces = 0;

    y.dp = 0.0f;
    y.dVC = 0.0f;

    return true;
}

bool TraceNewVertex(inout SampleGenerator sg, inout LightSubpathVertex yj, out bool rough)
{
    if (yj.numVertices == 1)
    {
        EmissiveTriangle lightTriangle = gScene.lightCollection.getTriangle(yj.y0.triangleIndex);
        float3 lightPos = lightTriangle.getPosition(yj.y0.bary);
        float3 lightNormal = lightTriangle.normal;

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

        TriangleHit y_s1_hit = hit.getTriangleHit();
        VertexData y_s1_vd = gScene.getVertexData(y_s1_hit);
        uint y_s1_materialID = gScene.getMaterialID(y_s1_hit.instanceID);
        ShadingData y_s1_sd = gScene.materials.prepareShadingData(y_s1_vd, y_s1_materialID, -worldDir);

        let y_s1_mi = gScene.materials.getMaterialInstance(y_s1_sd, ExplicitLodTextureSampler(0.f));
        BSDFProperties y_s1_hitProps = y_s1_mi.getProperties(y_s1_sd);
        float rgh = y_s1_hitProps.roughness;

        BSDFSample y_s1_bsdfSample;
        if (!y_s1_mi.sample(y_s1_sd, sg, y_s1_bsdfSample))
            return false;

        rough = rgh > gRoughnessThreshold;

        PathVertex y_s1;
        y_s1.instanceID = y_s1_hit.instanceID;
        y_s1.materialID = gScene.getMaterialID(y_s1_hit.instanceID);
        y_s1.posW = y_s1_vd.posW;
        y_s1.primitiveID = y_s1_hit.primitiveIndex;
        y_s1.dirIn = -ray.dir;
        y_s1.barycentrics = y_s1_hit.barycentrics;

        yj.y_s1 = y_s1;

        float2 uv = lightTriangle.getTexCoord(yj.y0.bary);
        float3 Le = gScene.materials.evalEmissive(lightTriangle.materialID, uv);

        yj.throughput = Le * (M_PI / yj.subpathPdf);

        float cosThetaIn = abs(dot(y_s1_vd.normalW, -worldDir));
        float cosThetaOut = abs(dot(lightNormal, worldDir));
        float distSq = max(hitT * hitT, 1e-7f);
        float pdfFwdArea = pdfSA * (cosThetaIn / distSq);

        float p0_trace = yj.subpathPdf;
        float p0_connect = yj.subpathPdf;

        yj.dp = pow(1.0f / pdfFwdArea, gMISPower);
        yj.dVC = pow((cosThetaOut / distSq) / pdfFwdArea, gMISPower);

        yj.subpathPdf *= pdfFwdArea;
    }
    else
    {
        TriangleHit y_s2_hit;
        y_s2_hit.instanceID = yj.y_s1.instanceID;
        y_s2_hit.primitiveIndex = yj.y_s1.primitiveID;
        y_s2_hit.barycentrics = yj.y_s1.barycentrics;

        VertexData y_s2_vd = gScene.getVertexData(y_s2_hit);
        ShadingData y_s2_sd = gScene.materials.prepareShadingData(y_s2_vd, yj.y_s1.materialID, yj.y_s1.dirIn);
        let y_s2_mi = gScene.materials.getMaterialInstance(y_s2_sd, ExplicitLodTextureSampler(0.f));

        BSDFSample y_s2_bsdfSample;
        if (!y_s2_mi.sample(y_s2_sd, sg, y_s2_bsdfSample))
            return false;

        SceneRayQuery<0> rq;
        Ray ray = Ray(y_s2_sd.posW, y_s2_bsdfSample.wo, 1e-4f, 10000.0f);
        float hitT;
        HitInfo hit = rq.traceRay(ray, hitT);

        if (!hit.isValid())
            return false;

        TriangleHit y_s1_hit = hit.getTriangleHit();
        VertexData y_s1_vd = gScene.getVertexData(y_s1_hit);
        uint y_s1_materialID = gScene.getMaterialID(y_s1_hit.instanceID);
        ShadingData y_s1_sd = gScene.materials.prepareShadingData(y_s1_vd, y_s1_materialID, -y_s2_bsdfSample.wo);

        let y_s1_mi = gScene.materials.getMaterialInstance(y_s1_sd, ExplicitLodTextureSampler(0.f));

        BSDFProperties y_s1_hitProps = y_s1_mi.getProperties(y_s1_sd);
        float rgh = y_s1_hitProps.roughness;

        rough = rgh > gRoughnessThreshold;

        float distSq = max(hitT * hitT, 1e-7f);
        float cosNext = abs(dot(y_s1_sd.frame.N, -y_s2_bsdfSample.wo));
        float pdfFwdArea = y_s2_bsdfSample.pdf * (cosNext / distSq);

        float3 dirToPrev = normalize(y_s2_sd.posW - y_s1_sd.posW);
        ShadingData sdNextReverse = gScene.materials.prepareShadingData(y_s1_vd, y_s1_materialID, -y_s2_bsdfSample.wo);
        let miNextReverse = gScene.materials.getMaterialInstance(sdNextReverse, ExplicitLodTextureSampler(0.f));
        float pdfRevSA = miNextReverse.evalPdf(sdNextReverse, dirToPrev);

        float old_dp = yj.dp;
        float old_dVC = yj.dVC;

        float invPdfFwdAreaPow = pow(1.0f / max(pdfFwdArea, 1e-30f), gMISPower);
        float pdfRevSAPow = pow(max(pdfRevSA, 0.0f), gMISPower);

        if (y_s2_bsdfSample.isLobe(LobeType::Delta))
        {
            yj.dp = old_dp;
            yj.dVC = old_dVC;
        }
        else
        {
            float invPdfFwdAreaPow = pow(1.0f / max(pdfFwdArea, 1e-30f), gMISPower);
            float pdfRevSAPow = pow(max(pdfRevSA, 0.0f), gMISPower);

            yj.dp = invPdfFwdAreaPow;
            yj.dVC = invPdfFwdAreaPow * (old_dp + pdfRevSAPow * old_dVC);
        }

        yj.throughput *= y_s2_bsdfSample.weight;
        yj.subpathPdf *= pdfFwdArea;

        yj.y_s1.instanceID = y_s1_hit.instanceID;
        yj.y_s1.materialID = y_s1_materialID;
        yj.y_s1.posW = y_s1_sd.posW;
        yj.y_s1.primitiveID = y_s1_hit.primitiveIndex;
        yj.y_s1.dirIn = -ray.dir;
        yj.y_s1.barycentrics = y_s1_hit.barycentrics;
    }

    yj.numBounces += (uint16_t)(rough ? 1 : 0);
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

    TriangleHit y_s1_hit;
    y_s1_hit.instanceID = yj.y_s1.instanceID;
    y_s1_hit.primitiveIndex = yj.y_s1.primitiveID;
    y_s1_hit.barycentrics = yj.y_s1.barycentrics;
    VertexData y_s1_vd = gScene.getVertexData(y_s1_hit);

    ShadingData y_s1_sd = gScene.materials.prepareShadingData(y_s1_vd, yj.y_s1.materialID, yj.y_s1.dirIn);
    let y_s1_mi = gScene.materials.getMaterialInstance(y_s1_sd, ExplicitLodTextureSampler(0.f));

    SceneRayQuery<0> rq;
    Ray ray = Ray(yj.y_s1.posW, dirToCamera, 1e-4f, dist - 1e-4f);

    if (!rq.traceVisibilityRay(ray))
        return false;

    bool isCaustic = (yj.numBounces > 0) && (yj.numVertices > 1);
    InitializeReservoir(isCaustic, r);

    float3 bsdf = y_s1_mi.eval(y_s1_sd, dirToCamera, sg);

    float cosThetaOut = abs(dot(y_s1_sd.frame.N, dirToCamera));
    float cosThetaCam = dot(normalize(gScene.camera.data.cameraW), -dirToCamera);

    float geometryFactor = (cosThetaOut * cosThetaCam) / distSq;

    float3 pathF = yj.throughput * bsdf * geometryFactor;

    r.pathF = pathF;

    float halfWidth = length(gScene.camera.data.cameraU);
    float halfHeight = length(gScene.camera.data.cameraV);
    float imagePlaneArea = 4.0f * halfWidth * halfHeight;

    float p_camSA = 1.0f / (imagePlaneArea * pow(cosThetaCam, 3.0f));
    float p_camArea = p_camSA * (cosThetaOut / distSq);

    ShadingData sdReverse = gScene.materials.prepareShadingData(y_s1_vd, yj.y_s1.materialID, dirToCamera);
    let miReverse = gScene.materials.getMaterialInstance(sdReverse, ExplicitLodTextureSampler(0.f));
    float pdfRevSA = miReverse.evalPdf(sdReverse, yj.y_s1.dirIn);
    float p_revArea = pdfRevSA * (cosThetaCam / distSq);

    float pathRatio = (gTargetDim.x * gTargetDim.y) / float(gNumLightSubpaths);

    float prefix = pow(p_camArea * pathRatio, gMISPower);
    float w_light = prefix * (yj.dp + pow(pdfRevSA, gMISPower) * yj.dVC);
    float misWeight = 1.0f / (1.0f + w_light);

    r.misWeight = misWeight;

    r.UCW = 1.0f;

    r.confidenceWeight = 1.0f;

    r.cameraSubpathSeed = 0;
    r.cameraSubpathId = 0;

    r.bounces = yj.numVertices - 1;

    r.prefixBounces = 0;
    r.prefixDiffuseBounces = 0;

    r.flags.setCaustic(isCaustic);

    r.prefixPdf = 1.0f;

    r.xr = yj.y_s1;

    r.suffixF = float3(0.0f);
    r.suffixPdf = 0.0f;

    r.reconnectionCos = cosThetaOut;
    r.reconnectionDist = sqrt(distSq);

    r.dirOut = dirToCamera;
    r.y0 = yj.y0;

    r.lightPathSeed = yj.subpathSeed;
    r.lightPathId = yj.subpathId;
    r.lightPathPdf = yj.subpathPdf;
    r.lightPathVertices = yj.numVertices;
    r.lightPathDiffuseBounces = yj.numBounces;

    MisEvalData misData = {};

    misData.d_p_light = yj.dp;
    misData.d_vc_light = yj.dVC;
    misData.p_rev_light = pdfRevSA * (cosThetaCam / distSq);

    misData.p_rev_sigma_light = pdfRevSA;

    misData.p0_trace_cam = 1.0f;
    misData.p0_connect_cam = 1.0f;

    r.cachedMisData = misData;

    return true;
}
