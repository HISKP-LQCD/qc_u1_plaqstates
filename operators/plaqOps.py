import numpy as np
import scipy.special
import scipy.integrate

def _getPlaqStateOps(l_max, g2):

    q = -1.0 / (g2 * g2)

    states = []
    stateDiff = []
    evals = []

    for n in range(2 * l_max + 1):

        if n % 2 == 1:
            a = scipy.special.mathieu_b(n + 1, q)
            states.append(lambda x, n=n: scipy.special.mathieu_sem(
                n + 1, q, x * 180 / np.pi)[0] / np.sqrt(np.pi))
            stateDiff.append(lambda x, n=n: 0.5 * scipy.special.mathieu_sem(
                n + 1, q, x * 180 / np.pi)[1] / np.sqrt(np.pi))
        else:
            a = scipy.special.mathieu_a(n, q)
            states.append(lambda x, n=n: scipy.special.mathieu_cem(
                n, q, x * 180 / np.pi)[0] / np.sqrt(np.pi))
            stateDiff.append(lambda x, n=n: 0.5 * scipy.special.mathieu_cem(
                n, q, x * 180 / np.pi)[1] / np.sqrt(np.pi))

        eval = 0.5 * g2 * a + 1.0 / g2
        evals.append(eval)

    ImTrP = np.zeros((2 * l_max + 1, 2 * l_max + 1))
    L = np.zeros((2 * l_max + 1, 2 * l_max + 1), dtype=np.complex128)

    for i in range(2 * l_max + 1):
        for j in range(2 * l_max + 1):

            L[i, j] = -1j * 2 * scipy.integrate.quad(
                lambda x: states[i](x) * stateDiff[j](x), 0, np.pi)[0]

            ImTrP[i, j] = -2 * 2 * scipy.integrate.quad(
                lambda x: states[i](x) * np.cos(2 * x) * states[j]
                (x), 0, np.pi)[0]

    ImTrP += 2.0 * np.identity(2 * l_max + 1)

    Lsq = (np.diag(evals) - 0.5 / g2 * ImTrP) / g2 / 2.0

    ImTrP[np.abs(ImTrP) < 1e-10] = 0
    Lsq[np.abs(Lsq) < 1e-10] = 0
    L[np.abs(L) < 1e-10] = 0

    return L, Lsq, ImTrP