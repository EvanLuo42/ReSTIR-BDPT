import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Raytracing;
import Scene.RaytracingInline;

struct PathVertex
{
    float3 pos;
    float3 normal;
    float3 faceNormal;

    /** Create a path vertex.
        \param[in] index Vertex index.
        \param[in] pos Vertex position.
        \param[in] normal Shading normal.
        \param[in] faceNormal Geometry normal.
        \return Returns a path vertex.
    */
    static PathVertex create(float3 pos, float3 normal = float3(0.f), float3 faceNormal = float3(0.f))
    {
        PathVertex v = {};
        v.pos = pos;
        v.normal = normal;
        v.faceNormal = faceNormal;
        return v;
    }

    /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
        for visibility rays.
        \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
        \return Returns the offseted position.
    */
    float3 getRayOrigin(float3 rayDir) { return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal); }
};

struct SceneVertex
{
    uint instanceID;
    uint primitiveID;
    float2 bary;
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
    ShadingData sd;
    IMaterialInstance mi;
    bool valid;
    bool isEmissive;
    bool isRough;
    uint matID;
    uint instanceID;
    uint primitiveID;
    float2 bary;
};

struct PathReservoirFlags
{
    uint flags;

    static const uint NONE = 0u;
    static const uint CAUSTIC = 1u << 0;
    static const uint FROM_LVC = 1u << 1;
    static const uint FROM_NEE = 1u << 2;
    static const uint FROM_EMISSION = 1u << 3;
    static const uint TEMPORAL = 1u << 4;
    static const uint SPATIAL = 1u << 5;
    static const uint DELTA_SEGMENT = 1u << 6;

    [mutating]
    void clear() { flags = 0u; }

    bool has(uint bit) { return (flags & bit) != 0u; }

    [mutating]
    void set(uint bit) { flags |= bit; }

    [mutating]
    void unset(uint bit) { flags &= ~bit; }

    bool isCaustic() { return has(CAUSTIC); }

    bool isFromLVC() { return has(FROM_LVC); }

    bool isFromNEE() { return has(FROM_NEE); }

    bool isFromEmission() { return has(FROM_EMISSION); }

    bool isTemporal() { return has(TEMPORAL); }

    bool isSpatial() { return has(SPATIAL); }

    bool hasDeltaSegment() { return has(DELTA_SEGMENT); }

    [mutating]
    void markCaustic() { set(CAUSTIC); }

    [mutating]
    void markFromLVC() { set(FROM_LVC); }

    [mutating]
    void markFromNEE() { set(FROM_NEE); }

    [mutating]
    void markFromEmission() { set(FROM_EMISSION); }

    [mutating]
    void markTemporal() { set(TEMPORAL); }

    [mutating]
    void markSpatial() { set(SPATIAL); }

    [mutating]
    void markDeltaSegment() { set(DELTA_SEGMENT); }
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
