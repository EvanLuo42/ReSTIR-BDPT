import PathReservoir;
import Scene.Scene;
import Scene.Raytracing;
import Scene.RaytracingInline;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;
import LightUtils;

#include "LRM.slangh"
#include "SpatiotemporalReuseCommon.slangh"

cbuffer CameraPathCB
{
    uint gFrameCount;
    uint gMaxBounces;
    uint2 gRenderTargetDim;

    uint gLvcCandidatesPerVertex;
    uint gNeeCandidatesPerVertex;
}

StructuredBuffer<LightSubpathVertex> gLightVertexCache;

StructuredBuffer<uint> gPerPixelLTCount;
StructuredBuffer<uint> gPerPixelLTOffset;
StructuredBuffer<PathReservoir> gPerPixelLTReservoirBuffer;

RWStructuredBuffer<PathReservoir> gOutputReservoir;
RWStructuredBuffer<PathReservoir> gOutputCausticReservoir;

void InitializeReservoir(out PathReservoir r)
{
    r.UCW = 0.0;
    r.misWeight = 1.0;
    r.confidenceWeight = 1.0;
    r.flags.clear();
}

bool TraceNewCameraVertex(Ray ray, out CameraSubpathVertex cv, out float hitT)
{
    cv.valid = false;

    SceneRayQuery<0> rq;
    HitInfo hit = rq.traceRay(ray, hitT);
    if (!hit.isValid())
        return false;

    TriangleHit triHit = hit.getTriangleHit();
    VertexData vd = gScene.getVertexData(triHit);

    cv.instanceID = triHit.instanceID.index;
    cv.primitiveID = triHit.primitiveIndex;
    cv.bary = triHit.barycentrics;

    cv.x = PathVertex::create(vd.posW, vd.normalW, vd.faceNormalW);

    cv.matID = gScene.getMaterialID(triHit.instanceID);
    cv.sd = gScene.materials.prepareShadingData(vd, cv.matID, -ray.dir);

    let lod = ExplicitLodTextureSampler(0.f);
    cv.mi = gScene.materials.getMaterialInstance(cv.sd, lod);

    float2 uv = vd.texC;
    float3 Le = gScene.materials.evalEmissive(cv.matID, uv);
    cv.isEmissive = any(Le > 0.0);

    cv.isRough = (cv.mi.getLobeTypes(cv.sd) & (uint)LobeType::NonDelta) != 0;

    cv.valid = true;

    return true;
}

bool ConnectToRandomLight(
    PathVertex xj,
    ShadingData sd,
    IMaterialInstance mi,
    float3 camThroughput,
    float camPrefixPdf,
    uint camBounces,
    inout SampleGenerator sg,
    out PathReservoir outCand,
    bool isDelta
)
{
    outCand = {};
    outCand.misWeight = 1.0;

    float selPdf;
    uint triIdx = SampleEmissiveTriangle(sg, selPdf);

    EmissiveTriangle tri = gScene.lightCollection.getTriangle(triIdx);
    MeshLightData meshData = gScene.lightCollection.meshData[tri.lightIdx];

    float3 bary = sample_triangle(sampleNext2D(sg));
    float3 yPos = tri.getPosition(bary);
    float3 yNg = tri.normal;
    float3 yNs = tri.normal;

    float2 uv = tri.getTexCoord(bary);
    float3 Le = gScene.materials.evalEmissive(tri.materialID, uv);

    float pA_y = selPdf / tri.area;

    float3 dir = normalize(yPos - xj.pos);
    float dist = length(yPos - xj.pos);

    SceneRayQuery<0> rq;
    Ray shadow = Ray(computeRayOrigin(xj.pos, xj.faceNormal), dir, 1e-4, dist - 1e-4);

    if (!rq.traceVisibilityRay(shadow))
        return false;

    float cosX = max(0.0, dot(xj.normal, dir));
    float cosY = max(0.0, dot(yNs, -dir));
    if (cosX <= 0.0 || cosY <= 0.0)
        return false;

    float3 f = mi.eval(sd, dir, sg);
    float pdfW = mi.evalPdf(sd, dir);
    if (pdfW <= 0.0)
        return false;

    float pW_light = pA_y * (dist * dist) / max(1e-8, cosY);

    float3 contrib = camThroughput * f * Le * (cosX / max(1e-8, pW_light));

    PathReservoir r = {};
    r.pathF = contrib;
    r.prefixPdf = camPrefixPdf;
    r.xr = xj;
    r.bounces = camBounces;

    r.flags.clear();
    r.flags.markFromNEE();

    if (isDelta)
        r.flags.markCaustic();

    r.UCW = 0.0;

    r.y0.instanceID = meshData.instanceID;
    r.y0.primitiveID = triIdx - meshData.triangleOffset;
    r.y0.bary = bary.yz;

    outCand = r;
    return true;
}

bool ConnectToRandomLVC(
    PathVertex xj,
    ShadingData sd,
    IMaterialInstance mi,
    float3 camThroughput,
    float camPrefixPdf,
    uint camBounces,
    inout SampleGenerator sg,
    out PathReservoir outCand,
    bool isDelta
)
{
    outCand = {};
    outCand.misWeight = 1.0;

    uint cell = LRM_ComputeCellIndex(xj.pos, gCellSize);
    if (cell == 0xFFFFFFFFu)
        return false;

    uint begin = gLRMCellOffset[cell];
    uint end = gLRMCellOffset[cell + 1];
    uint count = (end > begin) ? (end - begin) : 0;
    if (count == 0)
        return false;

    uint pick = begin + (uint)(sampleNext1D(sg) * count);
    pick = min(pick, end - 1);

    uint lvcIdx = gLRMCellStorage[pick];
    LightSubpathVertex y = gLightVertexCache[lvcIdx];

    float3 yPos = y.y_s1.pos;

    float3 dir = normalize(yPos - xj.pos);
    float dist = length(yPos - xj.pos);

    SceneRayQuery<0> rq;
    Ray shadow = Ray(computeRayOrigin(xj.pos, xj.faceNormal), dir, 1e-4, dist - 1e-4);
    if (!rq.traceVisibilityRay(shadow))
        return false;

    float cosX = max(0.0, dot(xj.normal, dir));
    float cosY = max(0.0, dot(y.y_s1.normal, -dir));
    if (cosX <= 0.0 || cosY <= 0.0)
        return false;

    float3 f = mi.eval(sd, dir, sg);
    float pdfW = mi.evalPdf(sd, dir);
    if (pdfW <= 0.0)
        return false;

    float G = (cosX * cosY) / max(1e-8, dist * dist);

    float3 contrib = camThroughput * f * y.throughput * G;

    PathReservoir r = {};
    r.pathF = contrib;
    r.prefixPdf = camPrefixPdf;
    r.xr = xj;
    r.bounces = camBounces;

    r.y0 = y.y0;

    r.lightPathSeed = y.subpathSeed;
    r.lightPathId = y.subpathId;
    r.lightPathPdf = y.subpathPdf;
    r.lightPathVertices = y.numVertices;

    r.flags.clear();
    r.flags.markFromLVC();

    if (isDelta)
        r.flags.markCaustic();

    r.UCW = 0.0;

    outCand = r;
    return true;
}

bool EvalEmissionHit(CameraSubpathVertex cv, float3 camThroughput, float camPrefixPdf, uint camBounces, out PathReservoir outCand, bool hasDelta)
{
    outCand = {};
    outCand.misWeight = 1.0;

    float3 Le = gScene.materials.evalEmissive(cv.matID, cv.sd.uv);

    if (cv.isEmissive && all(Le == 0.0))
        Le = float3(1.0);

    if (all(Le == 0.0))
        return false;

    PathReservoir r = {};
    r.pathF = camThroughput * Le;
    r.prefixPdf = camPrefixPdf;
    r.xr = cv.x;
    r.bounces = camBounces;

    r.flags.clear();
    r.flags.markFromEmission();

    if (hasDelta)
        r.flags.markCaustic();

    r.UCW = 0.0;

    outCand = r;
    return true;
}

[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= gRenderTargetDim.x || tid.y >= gRenderTargetDim.y)
        return;

    uint pixelIndex = tid.y * gRenderTargetDim.x + tid.x;

    SampleGenerator sg = SampleGenerator(tid.xy, gFrameCount);

    PathReservoir ri;
    InitializeReservoir(ri);
    PathReservoir ci; // Caustic reservoir
    InitializeReservoir(ci);

    Ray primaryRay = gScene.camera.computeRayPinhole(tid.xy, gRenderTargetDim);

    float3 throughput = float3(1.0);
    float prefixPdf = 1.0;

    float3 rayOrigin = primaryRay.origin;
    float3 rayDir = primaryRay.dir;

    bool hasDelta = false;

    // Camera tracing loop
    for (uint j = 1; j <= gMaxBounces; ++j)
    {
        Ray ray = Ray(rayOrigin, rayDir, 1e-4, 1e38);

        float hitT = 0.0;
        CameraSubpathVertex cv;
        TraceNewCameraVertex(ray, cv, hitT);
        if (!cv.valid)
            break;

        // s=0 technique
        if (cv.isEmissive)
        {
            PathReservoir rEm;
            if (EvalEmissionHit(cv, throughput, prefixPdf, j - 1, rEm, hasDelta))
            {
                ReservoirMerge(ri, rEm, TargetWeight(rEm.pathF), sg);
            }
        }

        // Connection strategies if xj is rough
        if (cv.isRough)
        {
            // (s=1) NEE candidates
            [loop]
            for (uint k = 0; k < max(1u, gNeeCandidatesPerVertex); ++k)
            {
                PathReservoir rNee;
                if (ConnectToRandomLight(cv.x, cv.sd, cv.mi, throughput, prefixPdf, j - 1, sg, rNee, hasDelta))
                {
                    ReservoirMerge(ri, rNee, TargetWeight(rNee.pathF), sg);
                }
            }

            // (s>=2) Random LVC candidates
            [loop]
            for (uint k = 0; k < max(1u, gLvcCandidatesPerVertex); ++k)
            {
                PathReservoir rLvc;
                if (ConnectToRandomLVC(cv.x, cv.sd, cv.mi, throughput, prefixPdf, j - 1, sg, rLvc, hasDelta))
                {
                    ReservoirMerge(ri, rLvc, TargetWeight(rLvc.pathF), sg);
                }
            }
        }

        BSDFSample bs;
        if (!cv.mi.sample(cv.sd, sg, bs, true))
            break;
        if (bs.pdf <= 0.0)
            break;

        if (!bs.isLobe(LobeType::NonDelta))
            hasDelta = true;

        throughput *= bs.weight;
        prefixPdf *= bs.pdf;

        rayDir = bs.wo;
        rayOrigin = cv.x.getRayOrigin(rayDir);
    }

    uint ltCount = gPerPixelLTCount[pixelIndex];
    uint ltBegin = gPerPixelLTOffset[pixelIndex];

    [loop]
    for (uint t = 0; t < ltCount; ++t)
    {
        PathReservoir rLT = gPerPixelLTReservoirBuffer[ltBegin + t];

        if (rLT.flags.isCaustic())
        {
            ReservoirMerge(ci, rLT, TargetWeight(rLT.pathF), sg);
        }
        else
        {
            ReservoirMerge(ri, rLT, TargetWeight(rLT.pathF), sg);
        }
    }

    gOutputReservoir[pixelIndex] = ri;
    gOutputCausticReservoir[pixelIndex] = ci;
}
