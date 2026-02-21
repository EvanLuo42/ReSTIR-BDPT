import LRM;

ParameterBlock<LRMData> gLRM;

[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint totalAppends = gLRM.appendCounter[0];
    if (tid.x >= totalAppends || tid.x >= gLRM.maxLightPaths)
        return;

    LRMAppendData data = gLRM.appendBuffer[tid.x];

    uint baseOffset = gLRM.indexBuffer[data.pixelIndex];

    uint scatterOffset = baseOffset + data.localIndex;

    gLRM.cellStorage[scatterOffset] = data.reservoirId;
}
