import PathReservoir;

cbuffer CB
{
    uint2 gTargetDim;
}

StructuredBuffer<PathReservoir> gOutputReservoirs;
StructuredBuffer<PathReservoir> gOutputCausticReservoirs;

RWTexture2D<float4> gOutputColor;

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixelCoord = tid.xy;

    if (pixelCoord.x >= gTargetDim.x || pixelCoord.y >= gTargetDim.y)
        return;

    uint pixelIndex = pixelCoord.y * gTargetDim.x + pixelCoord.x;

    PathReservoir r = gOutputReservoirs[pixelIndex];
    PathReservoir c = gOutputCausticReservoirs[pixelIndex];

    float3 finalColor = float3(0.0f, 0.0f, 0.0f);

    if (r.UCW > 0.0f)
    {
        finalColor += r.pathF * r.UCW;
    }

    if (c.UCW > 0.0f)
    {
        finalColor += c.pathF * c.UCW;
    }

    if (any(isnan(finalColor)) || any(isinf(finalColor)))
    {
        finalColor = float3(0.0f, 0.0f, 0.0f);
    }

    gOutputColor[pixelCoord] = float4(finalColor, 1.0f);
}
