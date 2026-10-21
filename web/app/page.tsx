import { HeroScene } from "../components/hero-scene";

export default function HomePage() {
  return (
    <>
      <p className="kicker">C++20 çerçevesi</p>
      <div className="grid">
        <div>
          <h1>Nexus, ölçülen işi yapan bir öğrenme çerçevesidir.</h1>
          <p className="lead">
            MatrixFlash-Pro, NexusData, NexusLoss, NexusOptim, NexusModel ve NexusTrain tek CMake hedefine bağlanır.
            Python kabuğu aynı adları kullanır. PyTorch taklidi yoktur.
          </p>
        </div>
        <HeroScene />
      </div>
      <h2>Bu makinede önde olan satırlar</h2>
      <p>
        21 Ekim 2026 akşamı, Ryzen 7 5800H, CPU, iş parçacığı etiketi 8. Depthwise 8.95–12.1 GFLOPS, PyTorch 4.93 ve
        TensorFlow 0.688. Attention en düşük Nexus turu 6870 çağrı/s, PyTorch 1192. LSTM, GRU, MSE ve Adam aynı sütunda
        Nexus&apos;un en düşük turu rakiplerin önündedir.
      </p>
      <h2 className="behind">Bu makinede geride olan satırlar</h2>
      <p>
        GEMM N=2048: Nexus 330, 362 ve 369 GFLOPS, TensorFlow 389. Üç tur da TensorFlow&apos;un altındadır. Conv2d: Nexus
        4.90–6.78 GFLOPS, PyTorch 31.6, TensorFlow 12.7, JAX 7.55. Dünya sıralaması çıkarılmadı.
      </p>
      <div className="grid">
        <article className="card">
          <strong className="ahead">Depthwise</strong>
          Aynı akşam PyTorch ve TensorFlow&apos;un önünde.
        </article>
        <article className="card">
          <strong className="behind">Conv2d CPU</strong>
          PyTorch 31.6 GFLOPS, Nexus 6.78&apos;e kadar.
        </article>
      </div>
    </>
  );
}
