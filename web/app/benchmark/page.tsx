export default function BenchmarkPage() {
  return (
    <>
      <p className="kicker">21 Ekim 2026 akşam</p>
      <h1>Aynı sütun, bu makine.</h1>
      <p>
        Kaynak <code>docs/benchmarks/2026-10-21-04-karsilastirma.md</code>. CPU, iş parçacığı etiketi 8. Nexus havuzu 16
        işçidir. Sayılar o akşamın JSON dosyalarındadır.
      </p>
      <table>
        <thead>
          <tr>
            <th>İş</th>
            <th>Nexus</th>
            <th>PyTorch</th>
            <th>TensorFlow</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>GEMM 2048 GFLOPS</td>
            <td>362 ve 369 (sıcak tur 330)</td>
            <td>336</td>
            <td>389</td>
          </tr>
          <tr>
            <td>Conv2d GFLOPS</td>
            <td>5.43 ve 6.78</td>
            <td>31.6</td>
            <td>12.7</td>
          </tr>
          <tr>
            <td>Depthwise</td>
            <td>10.4 ve 12.1</td>
            <td>4.93</td>
            <td>0.688</td>
          </tr>
          <tr>
            <td>Attention çağrı/s</td>
            <td>9653 ve 6870</td>
            <td>1192</td>
            <td>1076</td>
          </tr>
          <tr>
            <td>Adam GB/s</td>
            <td>28.2 ve 24.8</td>
            <td>4.71</td>
            <td>3.23</td>
          </tr>
        </tbody>
      </table>
      <h2>CUDA conv kapısı</h2>
      <p>
        Nexus CPU 6.39 GFLOPS, aygıtta kalan çekirdek 48.7, host gidiş-dönüş 19.0. Rakip satırı değildir. cuDNN kapalıdır.
      </p>
    </>
  );
}
