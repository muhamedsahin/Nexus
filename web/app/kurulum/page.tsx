import { CopyBlock } from "../../components/copy-block";

const cpp = `call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Preview\\VC\\Auxiliary\\Build\\vcvars64.bat"
"C:\\Program Files\\CMake\\bin\\cmake.exe" -S . -B build-phase1 -G Ninja -DCMAKE_BUILD_TYPE=Release
"C:\\Program Files\\CMake\\bin\\cmake.exe" --build build-phase1 --parallel`;

const py = `call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Preview\\VC\\Auxiliary\\Build\\vcvars64.bat"
"C:\\Program Files\\CMake\\bin\\cmake.exe" -S . -B build-phase1 -DNEXUS_BUILD_PYTHON=ON -DPython3_EXECUTABLE=C:\\ProgramData\\anaconda3\\python.exe
"C:\\Program Files\\CMake\\bin\\cmake.exe" --build build-phase1 --target _core --parallel
set PYTHONPATH=python\\src`;

export default function InstallPage() {
  return (
    <>
      <p className="kicker">Kurulum</p>
      <h1>Önce derleyici, sonra kabuk.</h1>
      <h2>C++</h2>
      <p>
        Windows, Visual Studio 2022, CMake 3.24 ve Ninja. CUDA 13.4 varsa cuBLAS GEMM ve <code>conv2d_device</code> açılır.
        cuDNN bu kurulumda yoktur ve kök CMake onu kapalı tutar. PATH üzerindeki cmake ESP-IDF olabilir; tam yol kullanın.
      </p>
      <CopyBlock code={cpp} />
      <h2>Python</h2>
      <p>Anaconda 3.11. ESP-IDF Python 3.14 kullanılmaz. Modül <code>python/src/nexus/_core.pyd</code> olarak yazılır.</p>
      <CopyBlock code={py} />
    </>
  );
}
