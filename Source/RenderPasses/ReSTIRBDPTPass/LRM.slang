import PathReservoir;

struct LRMAppendData
{
    uint pixelIndex;
    uint localIndex;
    uint reservoirId;
};

struct LRMData
{
    RWStructuredBuffer<PathReservoir> globalReservoirs;
    RWStructuredBuffer<uint> cellCounters;
    RWStructuredBuffer<LRMAppendData> appendBuffer;
    RWStructuredBuffer<uint> appendCounter;

    StructuredBuffer<uint> indexBuffer;
    RWStructuredBuffer<uint> cellStorage;

    uint maxLightPaths;
    uint numPixels;

    void Insert(uint pixelIndex, PathReservoir r)
    {
        if (pixelIndex >= numPixels)
            return;

        uint resId;
        InterlockedAdd(appendCounter[0], 1, resId);

        if (resId >= maxLightPaths)
            return;

        globalReservoirs[resId] = r;

        uint localIndex;
        InterlockedAdd(cellCounters[pixelIndex], 1, localIndex);

        LRMAppendData data = { pixelIndex, localIndex, resId };
        appendBuffer[resId] = data;
    }

    uint GetReservoirCount(uint pixelIndex) { return cellCounters[pixelIndex]; }

    PathReservoir GetReservoir(uint pixelIndex, uint indexInCell)
    {
        uint baseOffset = indexBuffer[pixelIndex];
        uint resId = cellStorage[baseOffset + indexInCell];
        return globalReservoirs[resId];
    }
};
