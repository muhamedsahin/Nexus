const rows = [
  ["nn.Linear / Conv2d / LSTM", "Katman. forward ve backward elle yazılmış geri geçiştir.", "model.add(nexus.nn.Linear(4, 2, True))"],
  ["nn.conv2d_device", "CUDA conv. Varsayılan Conv2d değildir.", "nexus.nn.conv2d_device(image, weight, bias, n, c, h, w, k, kh, kw)"],
  ["loss.CrossEntropy", "Logit ve sınıf indeksi. Gradyan çağıranın listesine yazılır.", "loss.forward_backward(logits, labels, classes)"],
  ["optim.AdamW / SGD", "Yüksek seviye güncelleme. Python'da büyük adım join eder.", "nexus.optim.AdamW(model, lr=0.05)"],
  ["train.TrainerBuilder", "fit, kayıp, metrik, zamanlayıcı, erken duruş.", "builder.adamw(model, lr=0.05).loss().build()"],
  ["data.load_image / save_image", "CHW float. Kayıt PNG, JPEG, BMP.", "nexus.data.save_image(path, chw, h, w, c)"],
  ["rl.PPOAgent", "Ayrık eylem. learn GIL'i bırakır.", "agent.learn(env, iterations, horizon)"],
  ["custom.Tape", "Kayıtlı işlemlerin gradyanı.", "tape.backward(output)"],
  [".nxs", "NXS2 model arşivi. NXS1 model değildir.", "nexus.load(path)"],
];

export default function ApiPage() {
  return (
    <>
      <p className="kicker">API</p>
      <h1>Ne işe yarar.</h1>
      {rows.map(([name, text, sample]) => (
        <article className="card" key={name} style={{ marginBottom: 12 }}>
          <strong>{name}</strong>
          <p>{text}</p>
          <code>{sample}</code>
        </article>
      ))}
      <p>
        Postgres, ODBC, Parquet ve tokenizer C++ NexusData API&apos;sidir. Python kabuğu bunları çağırmaz.
      </p>
    </>
  );
}
