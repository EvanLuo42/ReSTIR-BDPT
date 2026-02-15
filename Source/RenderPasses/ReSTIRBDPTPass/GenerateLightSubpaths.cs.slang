import PathReservoir;
import Scene.Scene;
import Scene.RaytracingInline;
import Scene.Shading;
import Scene.HitInfo;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Raytracing;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Utils.Math.MathHelpers;

#include "LRM.slangh"

cbuffer LightSubpathCB
{
    uint gNumLightSubpaths;
    uint gMaxBounces;
    uint gFrameCount;
    uint2 gRenderTargetDim;
    float gCellSize;
}

cbuffer EmissivePowerCB
{
    float gInvWeightsSum; // 1 / totalFlux
}

Buffer<uint2> gTriangleAliasTable;

RWStructuredBuffer<LightSubpathVertex> gLightVertexCache;

uint sampleEmissiveTriangle(inout SampleGenerator sg, out float selectionPdf)
{
    uint triangleCount = gScene.lightCollection.triangleCount;

    float uLight = sampleNext1D(sg);
    uint tableIndex = min((uint)(uLight * triangleCount), triangleCount - 1);

    uint2 packed = gTriangleAliasTable[tableIndex];
    float threshold = f16tof32(packed.x >> 16u);
    uint selectAbove = ((packed.x & 0xFFFFu) << 8u) | ((packed.y >> 24u) & 0xFFu);
    uint selectBelow = packed.y & 0xFFFFFFu;

    uint triIdx = (sampleNext1D(sg) >= threshold) ? selectAbove : selectBelow;

    selectionPdf = gScene.lightCollection.fluxData[triIdx].flux * gInvWeightsSum;
    return triIdx;
}

void initializeLightPath(
    inout SampleGenerator sg,
    out SceneVertex y0,
    out float3 y0PosW,
    out float3 y0NgW,
    out float3 y0NsW,
    out float3 Le,
    out float pA_y0
)
{
    float selectionPdf;
    uint triIdx = sampleEmissiveTriangle(sg, selectionPdf);

    EmissiveTriangle tri = gScene.lightCollection.getTriangle(triIdx);
    MeshLightData meshData = gScene.lightCollection.meshData[tri.lightIdx];

    float3 bary = sample_triangle(sampleNext2D(sg));

    y0.instanceID = meshData.instanceID;
    y0.primitiveID = triIdx - meshData.triangleOffset;
    y0.bary = bary.yz;

    y0PosW = tri.getPosition(bary);
    y0NgW = tri.normal;

    y0NsW = tri.normal;

    float2 uv = tri.getTexCoord(bary);
    Le = gScene.materials.evalEmissive(tri.materialID, uv);

    pA_y0 = selectionPdf / tri.area;
}

bool traceNewVertex(inout SampleGenerator sg, inout float3 rayOrigin, inout float3 rayDir, inout LightSubpathVertex v, out bool isRough)
{
    SceneRayQuery<0> rq;
    Ray ray = Ray(rayOrigin, rayDir, 1e-4f, 1e38f);

    float hitT;
    HitInfo hit = rq.traceRay(ray, hitT);

    if (!hit.isValid())
        return false;

    TriangleHit triHit = hit.getTriangleHit();
    VertexData vd = gScene.getVertexData(triHit);

    PathVertex pv = PathVertex::create(vd.posW, vd.normalW, vd.faceNormalW);
    v.y_s1 = pv;

    uint matID = gScene.getMaterialID(triHit.instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(vd, matID, -rayDir);

    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    BSDFSample bs;
    if (!mi.sample(sd, sg, bs, true))
        return false;

    if (bs.pdf <= 0.f)
        return false;

    isRough = bs.isLobe(LobeType::NonDelta);

    v.throughput *= bs.weight;
    v.subpathPdf *= bs.pdf;
    rayDir = bs.wo;

    v.numVertices++;
    v.numBounces++;

    return true;
}

bool connectToCamera(LightSubpathVertex v, out PathReservoir r, out uint pixelIndex)
{
    float3 x = v.y_s1.pos;

    float4 clip = mul(gScene.camera.getViewProj(), float4(x, 1));
    if (clip.w <= 0)
        return false;

    float2 ndc = clip.xy / clip.w;
    if (abs(ndc.x) > 1 || abs(ndc.y) > 1)
        return false;

    float2 screen = float2(ndc.x * 0.5f + 0.5f, -ndc.y * 0.5f + 0.5f);
    uint2 pixel = uint2(screen * gRenderTargetDim);
    pixelIndex = pixel.y * gRenderTargetDim.x + pixel.x;

    float3 camPos = gScene.camera.getPosition();
    float3 dir = normalize(camPos - x);
    float dist = length(camPos - x);

    SceneRayQuery<0> rq;
    Ray ray = Ray(computeRayOrigin(x, v.y_s1.faceNormal), dir, 1e-4, dist - 1e-4);

    if (!rq.traceVisibilityRay(ray))
        return false;

    float cosTheta = max(0, dot(v.y_s1.normal, dir));
    float G = cosTheta / (dist * dist);

    float3 contrib = v.throughput * G;

    r = {};
    r.pathF = contrib;
    r.prefixPdf = v.subpathPdf;

    return true;
}

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gNumLightSubpaths)
        return;

    SampleGenerator sg = SampleGenerator(uint2(i, 0), gFrameCount);

    SceneVertex y0 = {};
    float3 y0PosW, y0NgW, y0NsW, Le;
    float pA_y0;
    initializeLightPath(sg, y0, y0PosW, y0NgW, y0NsW, Le, pA_y0);

    LightSubpathVertex yj = {};
    yj.y0 = y0;

    yj.y_s1 = PathVertex::create(y0PosW, y0NsW, y0NgW);
    yj.numVertices = 1;
    yj.numBounces = 0;

    float pdfW0;
    float3 localDir = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdfW0);

    float3 T, B;
    branchlessONB(y0NsW, T, B);

    float3 dir0 = normalize(localDir.x * T + localDir.y * B + localDir.z * y0NsW);

    float cos0 = max(0.f, dot(y0NsW, dir0));

    yj.throughput = (Le * cos0) / (pA_y0 * pdfW0);
    yj.subpathPdf = pA_y0 * pdfW0;

    yj.subpathSeed = i;
    yj.subpathId = i;

    float3 rayDir = dir0;
    float3 rayOrigin = yj.y_s1.getRayOrigin(rayDir);

    for (uint j = 1; j < gMaxBounces; ++j)
    {
        bool isRough = false;
        bool alive = traceNewVertex(sg, rayOrigin, rayDir, yj, isRough);
        if (!alive)
            break;

        if (isRough)
        {
            gLightVertexCache[(i + 1) * j] = yj;
            PathReservoir r;
            uint k;

            if (connectToCamera(yj, r, k))
            {
                LRM_Insert(
                    yj.y_s1.pos,
                    yj.y_s1.normal,
                    r.pathF,
                    r.prefixPdf,
                    k,
                    yj.subpathSeed,
                    yj.numBounces,
                    gCellSize
                );
            }
        }
    }
}
