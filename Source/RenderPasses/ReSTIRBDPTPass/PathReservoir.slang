import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Raytracing;
import Scene.RaytracingInline;

struct PathVertex
{
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
    float d_p_cam;         // $$d_{t-1}^p$$
    float d_vc_cam;        // $$d_{t-1}^{VC}$$
    float p_fwd_cam;       // $$\vec{p}_{t-1}$$
    float p_rev_sigma_cam; // $$\vec{p}_{t-2} ^\sigma$$

    float d_p_light;   // $$d_{s-1}^p$$
    float d_vc_light;  // $$d_{s-1}^{VC}$$
    float p_rev_light; // $$\overleftarrow{p}_{s-1}$$
    float p_rev_sigma_light; // $$\overleftarrow{p}_{s-2}^\sigma$$

    float d_p_r;         // $$d_r^p$$
    float d_vc_r;        // $$d_r^{VC}$$
    float p_rev_sigma_r; // $$\overleftarrow{p}_{r-1}^\sigma$$
    float gamma;         // $$\vec{\gamma} = (g_{t-2} / p_{t-1})^\beta$$

    float lambda_vc; // $$\overleftarrow{\lambda}^{VC}$$

    float lambda_p; // $$\overleftarrow{\lambda}^p$$
    float sigma;    // $$\overleftarrow{\sigma}$$
    float g_ratio; // $$\overleftarrow{\sigma}$$

    float p0_trace_light;   // $$p_0^{trace}(\bar{y})$$
    float p0_connect_light; // $$p_0^{connect}(\bar{y})$$
    float p0_trace_cam;     // $$p_0^{trace}(\bar{z})$$
    float p0_connect_cam;   // $$p_0^{connect}(\bar{z})$$
    float p_rev_cam;        // $$\overleftarrow{p}_{t-1}$$
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
