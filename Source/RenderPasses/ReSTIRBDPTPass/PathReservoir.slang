import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Raytracing;
import Scene.RaytracingInline;

struct PathVertex {
    GeometryInstanceID instanceID;
    uint primitiveID;
    float2 barycentrics;

    float3 dirIn;
    float3 posW;
    uint materialID;
};

struct SceneVertex
{
    uint triangleIndex;
    float3 bary;
};

struct LightSubpathVertex
{
    SceneVertex y0;

    PathVertex y_s1;

    float3 throughput;
    float subpathPdf;

    uint subpathSeed;
    uint subpathId;

    uint diffuseBounces;

    uint16_t numVertices;
    uint16_t numBounces;

    float dVC;
    float dp;
}

struct MisEvalData
{
    float dVC;
    float dP;

    float gammaBar;
    float lambdaVCBar;
    float lambdaPBar;
    float sigmaBar;
};

struct CameraSubpathVertex
{
    PathVertex x;
};

struct PathReservoirFlags
{
    uint flags;

    static const uint CAUSTIC = 1u << 0;

    property bool IsCaustic
    {
        get {
            return (flags & CAUSTIC) != 0;
        }

        set {
            if (newValue)
                flags |= CAUSTIC;
            else
                flags &= ~CAUSTIC;
        }
    };
};

struct PathReservoir
{
    float UCW;
    float confidenceWeight;

    uint cameraSubpathSeed;
    uint cameraSubpathId;

    uint bounces;
    uint prefixBounces;
    uint prefixDiffuseBounces;
    PathReservoirFlags flags;

    float3 pathF;
    float misWeight;

    float prefixPdf;

    PathVertex xr;

    float3 suffixF;
    float suffixPdf;

    float reconnectionCos;
    float reconnectionDist;

    float3 dirOut;
    float pad2;

    SceneVertex y0;

    uint lightPathSeed;
    uint lightPathId;

    float lightPathPdf;
    uint lightPathVertices;

    uint lightPathDiffuseBounces;

    MisEvalData cachedMisData;
};
