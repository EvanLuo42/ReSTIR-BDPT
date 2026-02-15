#include "LRM.slangh"

[numthreads(256, 1, 1)]
void main(uint id: SV_DispatchThreadID)
{
    uint tripletCount = gLRMTripletCount[0];
    if (id >= tripletCount)
        return;

    LRMTriplet t = gLRMTriplets[id];

    if (t.localIndex >= gLRMMaxPerCell)
        return;

    uint base = gLRMCellOffset[t.cellEntryId];
    uint dst = base + t.localIndex;

    if (dst >= gLRMMaxCellStorage)
        return;

    gLRMCellStorage[dst] = t.recordId;
}
