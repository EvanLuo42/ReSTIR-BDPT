import Utils.Geometry.GeometryHelpers;

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

struct PathReservoirFlags
{
    uint flags = 0;
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
    uint flags;

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
    uint pad3;

    MisEvalData cachedMisData;
};
