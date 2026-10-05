import argparse
import numpy as np
import pywt
import matplotlib.pyplot as plt


def wavelet_anomalies(data, wavelet="db4", level=4, k=2.0):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745           
    thr = sigma * np.sqrt(2 * np.log(len(data)))                 
    den_coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    denoised = pywt.waverec(den_coeffs, wavelet)[:len(data)]
    residual = data - denoised
    idx = np.where(np.abs(residual) > k * np.std(residual))[0]
    return denoised, residual, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--wavelet", default="db4")
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--k", type=float, default=2.0, help="threshold = k * std(residual)")
    ap.add_argument("--out", default="task3_result.png")
    a = ap.parse_args()

    if a.csv:
        data = np.loadtxt(a.csv, delimiter=",", usecols=0)
    else:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(1000)
        if a.inject:
            for i in (150, 420, 700, 880):
                data[i] += rng.choice([-1, 1]) * 6
            data[500:520] += 3

    denoised, residual, idx = wavelet_anomalies(data, a.wavelet, a.level, a.k)
    print(f"Samples: {len(data)} | anomalies detected: {len(idx)}")
    print("Anomaly indices:", idx.tolist())

    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    ax[0].plot(data, label="Sensor Data")
    ax[0].plot(denoised, "--", label="Denoised Signal")
    ax[0].set_title("Sensor Data and Denoised Signal"); ax[0].legend(loc="lower right")
    ax[1].plot(residual, "r", label="Residuals")
    ax[1].scatter(idx, residual[idx], c="g", edgecolors="r", zorder=3, label="Anomalies")
    ax[1].set_title("Residuals and Detected Anomalies"); ax[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(a.out, dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
