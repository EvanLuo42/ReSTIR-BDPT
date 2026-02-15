#include "LRM.slangh"

[numthreads(256, 1, 1)]
void main(uint id: SV_DispatchThreadID)
{
    if (id == 0)
    {
        gLRMRecordCount[0] = 0;
        gLRMTripletCount[0] = 0;
    }

    if (id < LRM_CELL_ENTRY_COUNT)
    {
        gLRMCellChecksum[id] = 0;
        gLRMCellCount[id] = 0;
        gLRMCellOffset[id] = 0;
    }
}
