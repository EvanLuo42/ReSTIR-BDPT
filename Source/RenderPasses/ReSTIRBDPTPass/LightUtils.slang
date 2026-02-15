import Utils.Sampling.SampleGenerator;
import Scene;

uint SampleEmissiveTriangle(inout SampleGenerator sg, out float selectionPdf)
{
    uint triangleCount = gScene.lightCollection.triangleCount;
    float u = sampleNext1D(sg);
    uint triIdx = min((uint)(u * triangleCount), triangleCount - 1u);
    selectionPdf = 1.0f / max(1u, triangleCount);
    return triIdx;
}
