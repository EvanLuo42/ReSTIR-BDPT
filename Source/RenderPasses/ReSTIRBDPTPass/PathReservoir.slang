import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Raytracing;
import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;

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

    float d_p_light;         // $$d_{s-1}^p$$
    float d_vc_light;        // $$d_{s-1}^{VC}$$
    float p_rev_light;       // $$\overleftarrow{p}_{s-1}$$
    float p_rev_sigma_light; // $$\overleftarrow{p}_{s-2}^\sigma$$

    float d_p_r;         // $$d_r^p$$
    float d_vc_r;        // $$d_r^{VC}$$
    float p_rev_sigma_r; // $$\overleftarrow{p}_{r-1}^\sigma$$
    float gamma;         // $$\vec{\gamma} = (g_{t-2} / p_{t-1})^\beta$$

    float lambda_vc; // $$\overleftarrow{\lambda}^{VC}$$

    float lambda_p; // $$\overleftarrow{\lambda}^p$$
    float sigma;    // $$\overleftarrow{\sigma}$$
    float g_ratio;  // $$\overleftarrow{\sigma}$$

    float p0_trace_light;   // $$p_0^{trace}(\bar{y})$$
    float p0_connect_light; // $$p_0^{connect}(\bar{y})$$
    float p0_trace_cam;     // $$p_0^{trace}(\bar{z})$$
    float p0_connect_cam;   // $$p_0^{connect}(\bar{z})$$
    float p_rev_cam;        // $$\overleftarrow{p}_{t-1}$$
};

struct CameraSubpathVertex
{
    PathVertex x;

    float3 throughput;   // Accumulated path throughput
    uint bounces;        // Current depth/bounce count
    uint diffuseBounces; // Number of rough/diffuse bounces encountered
    float pathPdf;

    float dVC; // $$d_i^{VC}$$
    float dp;  // $$d_i^p$$

    float pdfFwd;      // Forward area-measure PDF: $$\vec{p}_i$$
    float pdfRevSigma; // Reverse solid-angle PDF: $$\overleftarrow{p}_i^\sigma$$

    float gamma;     // $$\vec{\gamma}$$
    float lambda_vc; // $$\overleftarrow{\lambda}^{VC}$$
    float lambda_p;  // $$\overleftarrow{\lambda}^p$$
    float sigma;     // $$\overleftarrow{\sigma}$$
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

    SceneVertex y0;

    uint lightPathSeed;
    uint lightPathId;

    float lightPathPdf;
    uint lightPathVertices;

    uint lightPathDiffuseBounces;

    MisEvalData cachedMisData;

    [mutating]
    void Merge(inout SampleGenerator sg, in PathReservoir r)
    {
        float q_hat = luminance(r.pathF);
        float p_hat = q_hat * r.misWeight;

        float w = p_hat * r.UCW;

        this.UCW += w;

        this.confidenceWeight += r.confidenceWeight;

        if (sampleNext1D(sg) * this.UCW < w)
        {
            this.cameraSubpathSeed = r.cameraSubpathSeed;
            this.cameraSubpathId = r.cameraSubpathId;

            this.bounces = r.bounces;
            this.prefixBounces = r.prefixBounces;
            this.prefixDiffuseBounces = r.prefixDiffuseBounces;
            this.flags = r.flags;

            this.pathF = r.pathF;
            this.misWeight = r.misWeight;
            this.prefixPdf = r.prefixPdf;
            this.xr = r.xr;
            this.suffixF = r.suffixF;
            this.suffixPdf = r.suffixPdf;
            this.reconnectionCos = r.reconnectionCos;
            this.reconnectionDist = r.reconnectionDist;
            this.dirOut = r.dirOut;
            this.y0 = r.y0;

            this.lightPathSeed = r.lightPathSeed;
            this.lightPathId = r.lightPathId;
            this.lightPathPdf = r.lightPathPdf;
            this.lightPathVertices = r.lightPathVertices;
            this.lightPathDiffuseBounces = r.lightPathDiffuseBounces;

            this.cachedMisData = r.cachedMisData;
        }
    }

    [mutating]
    void FinalizeResampling()
    {
        float q_hat = luminance(this.pathF);
        float p_hat = q_hat * this.misWeight;

        if (p_hat > 0.0f)
        {
            this.UCW = this.UCW / p_hat;
        }
        else
        {
            this.UCW = 0.0f;
        }
    }
};

void InitializeReservoir(bool isCaustic, out PathReservoir ri)
{
    ri.UCW = 0.0f;
    ri.confidenceWeight = 1.0f;

    ri.flags.flags = 0u;
    ri.flags.IsCaustic = isCaustic;

    ri.pathF = float3(0.0f, 0.0f, 0.0f);
    ri.misWeight = 0.0f;
    ri.prefixPdf = 0.0f;
    ri.suffixPdf = 0.0f;
    ri.lightPathPdf = 0.0f;
    ri.bounces = 0;
}
