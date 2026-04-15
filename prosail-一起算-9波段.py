import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc  # 引入拉丁超立方采样
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings

# 忽略优化过程中的警告，保持控制台整洁
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")

# ======================= 1. 配置区域 =======================
# 输入几何文件路径 (Step 1 气象与角度数据)
GEOMETRY_FILE = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\基于物理公式获取计算散射分数的模型的数据\step1_simulated_weather_data_spitters.csv"

# 🌟 核心联动：读取上一步 6S 模拟生成的 9 波段多项式系数文件
COEFFS_CSV_FILE = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\基于6s的公式系数\9_bands_coefficients.csv"

# 输出路径 (最终的综合训练集)
SAVE_PATH = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\Prosail-data\soil-select\weiss参数设置\特征约束\27核参数一起算\prosail_training_database_visionpoint.csv"

# 原始 PROSAIL 光谱参数文件
PROSAIL_CSV_FILE = r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\prosail配置文件\dataSpec_P5.csv"

# 🌟 本地化机理：引入三条南京白马基地实测土壤光谱库
CUSTOM_SOIL_FILES = [
    r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\白马土壤数据\24年小麦播种前裸土\PROSAIL_Trimmed_Soil_Background.csv",  # 50% 小麦旱地
    r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\白马土壤数据\25年水稻种植前泡田\PROSAIL_Trimmed_Soil_Background.csv",  # 35% 水稻早期水体
    r"E:\叶绿素反演-李文娟老师论文\原始论文中的波段\白马土壤数据\25年水稻播种前裸土\PROSAIL_Trimmed_Soil_Background.csv",  # 15% 水稻中后期湿泥
]

N_CASES = 50000
VZA_FIXED = 45.0
SENSOR_AZIMUTH = 90.0

# ======================= 2. 动态加载波段与系数 =======================
print(f"正在读取波段与 6S 多项式系数: {COEFFS_CSV_FILE}")
if not os.path.exists(COEFFS_CSV_FILE):
    raise FileNotFoundError("❌ 错误：找不到 6S 系数文件，请先运行 6S 模拟脚本！")

coeff_df = pd.read_csv(COEFFS_CSV_FILE)
# 提取 9 个波段名称 (如 '455nm')
BAND_NAMES = coeff_df['Band'].tolist()
# 提取波段数值
VISIONPOINT_BANDS = [int(name.replace('nm', '')) for name in BAND_NAMES]
# 确保波段索引对齐 PROSAIL 截断后的 440nm 起始位置
BAND_INDICES = [w - 440 for w in VISIONPOINT_BANDS]

# 动态构建 6S 多项式映射表
F_LAMBDA_COEFFS = {}
for _, row in coeff_df.iterrows():
    # 每一行包含：x^5, x^4, x^3, x^2, x^1, Intercept
    F_LAMBDA_COEFFS[row['Band']] = [
        row['x^5'], row['x^4'], row['x^3'], row['x^2'], row['x^1'], row['Intercept']
    ]

print(f"✅ 成功对接 6S 模拟结果，已载入波段: {VISIONPOINT_BANDS}")


# ======================= 3. PROSAIL 核心模型封装 =======================
class ProsailModel:
    def __init__(self, spec_csv_path, custom_soil_paths=None):
        if not os.path.exists(spec_csv_path):
            raise FileNotFoundError(f"找不到光谱配置文件: {spec_csv_path}")

        with open(spec_csv_path, 'r') as f:
            lines = f.readlines()

        data = np.array([[float(v) for v in line.strip().split(',')[1:]] for line in lines])
        wavs = data[0, :]

        # 统一按照 440-921nm 截取
        idx_start = np.where(wavs == 440)[0][0]
        idx_end = np.where(wavs == 921)[0][0] + 1
        self.spectra = data[:, idx_start:idx_end]
        target_len = idx_end - idx_start

        self.custom_soils = []
        if custom_soil_paths:
            for path in custom_soil_paths:
                if os.path.exists(path):
                    # 采用鲁棒读取逻辑，处理 UTF-8 签名和表头
                    df_soil = pd.read_csv(path, encoding='utf-8-sig')
                    col_idx = 0 if df_soil.shape[1] == 1 else 1
                    # 强制数值转换，防止非数字表头干扰
                    custom_soil = pd.to_numeric(df_soil.iloc[:, col_idx], errors='coerce').dropna().values

                    if np.max(custom_soil) > 1.0:
                        custom_soil = custom_soil / 100.0
                    if len(custom_soil) >= target_len:
                        self.custom_soils.append(custom_soil[:target_len])
                else:
                    raise FileNotFoundError(f"❌ 致命错误: 找不到本地土壤文件: {path}")

        if len(self.custom_soils) != 3:
            raise ValueError(f"❌ 致命错误: 必须提供准确的 3 条土壤光谱！")

    def _tav_abs(self, theta, refr):
        refr = np.array(refr)
        thetarad = np.radians(theta)
        res = np.zeros(refr.size)
        if theta == 0.:
            res = 4. * refr / (refr + 1.) ** 2
        else:
            refr2 = refr * refr
            ax = (refr + 1.) ** 2 / 2.
            bx = -(refr2 - 1.) ** 2 / 4.
            b1 = ((np.sin(thetarad) ** 2 - (refr2 + 1.) / 2.) ** 2 + bx) ** 0.5
            b2 = np.sin(thetarad) ** 2 - (refr2 + 1.) / 2.
            b0 = b1 - b2
            ts = (bx ** 2 / (6. * b0 ** 3) + bx / b0 - b0 / 2.) - (bx ** 2 / (6. * ax ** 3) + bx / ax - ax / 2.)
            tp1 = -2. * refr2 * (b0 - ax) / (refr2 + 1.) ** 2
            tp2 = -2. * refr2 * (refr2 + 1.) * np.log(b0 / ax) / (refr2 - 1.) ** 2
            tp3 = refr2 * (1. / b0 - 1. / ax) / 2.
            tp4 = 16. * refr2 ** 2 * (refr2 ** 2 + 1.) * np.log(
                (2. * (refr2 + 1.) * b0 - (refr2 - 1.) ** 2) / ((2. * (refr2 + 1.) * ax - (refr2 - 1.) ** 2))) / (
                          (refr2 + 1.) ** 3 * (refr2 - 1.) ** 2)
            tp5 = 16. * refr2 ** 3 * (1. / (2. * (refr2 + 1.) * b0 - ((refr2 - 1.) ** 2)) - 1. / (
                    2. * (refr2 + 1.) * ax - (refr2 - 1.) ** 2)) / (refr2 + 1.) ** 3
            tp = tp1 + tp2 + tp3 + tp4 + tp5
            res = (ts + tp) / (2. * np.sin(thetarad) ** 2)
        return res

    def _prospect_5B(self, N, Cab, Car, Cbrown, Cw, Cm):
        k = (Cab * self.spectra[2] + Car * self.spectra[3] + Cbrown * self.spectra[4] + Cw * self.spectra[5] + Cm *
             self.spectra[6]) / N
        refractive = self.spectra[1]
        tau, xx, yy = np.zeros(k.size), np.zeros(k.size), np.zeros(k.size)

        for i in range(tau.size):
            if k[i] <= 0.0:
                tau[i] = 1
            elif (k[i] > 0.0 and k[i] <= 4.0):
                xx[i] = 0.5 * k[i] - 1.0
                yy[i] = (((((((((((((((-3.60311230482612224e-13 * xx[i] + 3.46348526554087424e-12) * xx[
                    i] - 2.99627399604128973e-11) * xx[i] + 2.57747807106988589e-10) * xx[i] - 2.09330568435488303e-9) *
                                   xx[i] + 1.59501329936987818e-8) * xx[i] - 1.13717900285428895e-7) * xx[
                                     i] + 7.55292885309152956e-7) * xx[i] - 4.64980751480619431e-6) * xx[
                                   i] + 2.63830365675408129e-5) * xx[i] - 1.37089870978830576e-4) * xx[
                                 i] + 6.47686503728103400e-4) * xx[i] - 2.76060141343627983e-3) * xx[
                               i] + 1.05306034687449505e-2) * xx[i] - 3.57191348753631956e-2) * xx[
                             i] + 1.07774527938978692e-1) * xx[i] - 2.96997075145080963e-1
                yy[i] = (yy[i] * xx[i] + 8.64664716763387311e-1) * xx[i] + 7.42047691268006429e-1
                yy[i] = yy[i] - np.log(k[i])
                tau[i] = (1.0 - k[i]) * np.exp(-k[i]) + k[i] ** 2 * yy[i]
            elif (k[i] > 4.0 and k[i] <= 85.0):
                xx[i] = 14.5 / (k[i] + 3.25) - 1.0
                yy[i] = (((((((((((((((-1.62806570868460749e-12 * xx[i] - 8.95400579318284288e-13) * xx[
                    i] - 4.08352702838151578e-12) * xx[i] - 1.45132988248537498e-11) * xx[
                                        i] - 8.35086918940757852e-11) * xx[i] - 2.13638678953766289e-10) * xx[
                                      i] - 1.10302431467069770e-9) * xx[i] - 3.67128915633455484e-9) * xx[
                                    i] - 1.66980544304104726e-8) * xx[i] - 6.11774386401295125e-8) * xx[
                                  i] - 2.70306163610271497e-7) * xx[i] - 1.05565006992891261e-6) * xx[
                                i] - 4.72090467203711484e-6) * xx[i] - 1.95076375089955937e-5) * xx[
                              i] - 9.16450482931221453e-5) * xx[i] - 4.05892130452128677e-4) * xx[
                            i] - 2.14213055000334718e-3
                yy[i] = ((yy[i] * xx[i] - 1.06374875116569657e-2) * xx[i] - 8.50699154984571871e-2) * xx[
                    i] + 9.23755307807784058e-1
                yy[i] = np.exp(-k[i]) * yy[i] / k[i]
                tau[i] = (1.0 - k[i]) * np.exp(-k[i]) + k[i] ** 2 * yy[i]
            else:
                tau[i] = 0

        t1, t2 = self._tav_abs(90., refractive), self._tav_abs(40., refractive)
        x1 = 1 - t1
        x2 = t1 ** 2 * tau ** 2 * (refractive ** 2 - t1)
        x3 = t1 ** 2 * tau * refractive ** 2
        x4 = refractive ** 4 - tau ** 2 * (refractive ** 2 - t1) ** 2
        x5 = t2 / t1
        x6 = x5 * (t1 - 1) + 1 - t2
        r, t = x1 + x2 / x4, x3 / x4
        ra, ta = x5 * r + x6, x5 * t

        delta = (t ** 2 - r ** 2 - 1) ** 2 - 4 * r ** 2
        beta = (1 + r ** 2 - t ** 2 - delta ** 0.5) / (2 * r)
        va = (1 + r ** 2 - t ** 2 + delta ** 0.5) / (2 * r)
        vb = (beta * (va - r) / (va * (beta - r))) ** 0.5
        s1 = ra * (va * vb ** (N - 1) - va ** (-1) * vb ** (-(N - 1))) + (ta * t - ra * r) * (
                vb ** (N - 1) - vb ** (-(N - 1)))
        s2 = ta * (va - va ** (-1))
        s3 = va * vb ** (N - 1) - va ** (-1) * vb ** (-(N - 1)) - r * (vb ** (N - 1) - vb ** (-(N - 1)))
        return s1 / s3, s2 / s3

    def _campbell(self, n, ala):
        tx2 = np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88.])
        tx1 = np.array([10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88., 90.])
        tl1, tl2 = tx1 * np.arctan(1.) / 45., tx2 * np.arctan(1.) / 45.
        excent = np.exp(-1.6184e-5 * ala ** 3 + 2.1145e-3 * ala ** 2 - 1.2390e-1 * ala + 3.2491)
        x1 = excent / (np.sqrt(1. + excent ** 2 * np.tan(tl1) ** 2))
        x2 = excent / (np.sqrt(1. + excent ** 2 * np.tan(tl2) ** 2))
        if excent == 1.:
            freq = np.absolute(np.cos(tl1) - np.cos(tl2))
        else:
            alpha = excent / np.sqrt(np.absolute(1. - excent ** 2))
            alpha2, x12, x22 = alpha ** 2, x1 ** 2, x2 ** 2
            if excent > 1:
                alpx1, alpx2 = np.sqrt(alpha2 + x12), np.sqrt(alpha2 + x22)
                dum = x1 * alpx1 + alpha2 * np.log(x1 + alpx1)
                freq = np.absolute(dum - (x2 * alpx2 + alpha2 * np.log(x2 + alpx2)))
            else:
                almx1, almx2 = np.sqrt(alpha2 - x12), np.sqrt(alpha2 - x22)
                dum = x1 * almx1 + alpha2 * np.arcsin(x1 / alpha)
                freq = np.absolute(dum - (x2 * almx2 + alpha2 * np.arcsin(x2 / alpha)))
        return freq / np.sum(freq)

    def _volscatt(self, tts, tto, psi, ttl):
        rd = np.pi / 180.
        costs, costo, sints, sinto = np.cos(rd * tts), np.cos(rd * tto), np.sin(rd * tts), np.sin(rd * tto)
        cospsi, psir = np.cos(rd * psi), rd * psi
        costl, sintl = np.cos(rd * ttl), np.sin(rd * ttl)
        cs, co, ss, so = costl * costs, costl * costo, sintl * sints, sintl * sinto

        cosbts = -cs / ss if np.absolute(ss) > 1e-6 else 5.
        cosbto = -co / so if np.absolute(so) > 1e-6 else 5.

        if np.absolute(cosbts) < 1.:
            bts, ds = np.arccos(cosbts), ss
        else:
            bts, ds = np.pi, cs
        chi_s = 2. / np.pi * ((bts - np.pi * .5) * cs + np.sin(bts) * ss)

        if np.absolute(cosbto) < 1.:
            bto, doo = np.arccos(cosbto), so
        elif tto < 90.:
            bto, doo = np.pi, co
        else:
            bto, doo = 0, -co
        chi_o = 2. / np.pi * ((bto - np.pi * .5) * co + np.sin(bto) * so)

        btran1 = np.absolute(bts - bto)
        btran2 = np.pi - np.absolute(bts + bto - np.pi)

        if psir <= btran1:
            bt1, bt2, bt3 = psir, btran1, btran2
        else:
            bt1 = btran1
            if psir <= btran2:
                bt2, bt3 = psir, btran2
            else:
                bt2, bt3 = btran2, psir

        t1 = 2. * cs * co + ss * so * cospsi
        t2 = np.sin(bt2) * (2. * ds * doo + ss * so * np.cos(bt1) * np.cos(bt3)) if bt2 > 0. else 0.
        denom = 2. * np.pi * np.pi
        frho = ((np.pi - bt2) * t1 + t2) / denom
        ftau = (-bt2 * t1 + t2) / denom
        return chi_s, chi_o, max(frho, 0), max(ftau, 0)

    def _Jfunc1(self, k, l, t):
        d = (k - l) * t
        Jout = np.zeros(d.size)
        for i in range(Jout.size):
            if np.absolute(d[i]) > 1e-3:
                Jout[i] = (np.exp(-l[i] * t) - np.exp(-k * t)) / (k - l[i])
            else:
                Jout[i] = 0.5 * t * (np.exp(-k * t) + np.exp(-l[i] * t)) * (1. - d[i] * d[i] / 12.)
        return Jout

    def _Jfunc2(self, k, l, t):
        return (1. - np.exp(-(k + l) * t)) / (k + l)

    def _Jfunc3(self, k, l, t):
        return (1. - np.exp(-(k + l) * t)) / (k + l)

    def _PRO4SAIL(self, rho, tau, lidf, lai, q, tts, tto, psi, rsoil):
        litab = np.array([5., 15., 25., 35., 45., 55., 65., 75., 81., 83., 85., 87., 89.])
        rd = np.pi / 180.
        cts, cto = np.cos(rd * tts), np.cos(rd * tto)
        ctscto = cts * cto
        tants, tanto = np.tan(rd * tts), np.tan(rd * tto)
        cospsi = np.cos(rd * psi)
        dso = np.sqrt(tants * tants + tanto * tanto - 2. * tants * tanto * cospsi)

        # 🌟 核心修复：纯土壤反射的“黑洞 Bug”
        if lai <= 0: return rsoil, rsoil, rsoil, rsoil

        ks, ko, bf, sob, sof = 0, 0, 0, 0, 0
        ctl = np.cos(rd * litab)
        for i in range(13):
            chi_s, chi_o, frho, ftau = self._volscatt(tts, tto, psi, litab[i])
            ksli, koli = chi_s / cts, chi_o / cto
            sobli, sofli = frho * np.pi / ctscto, ftau * np.pi / ctscto
            bfli = ctl[i] * ctl[i]
            ks += ksli * lidf[i]
            ko += koli * lidf[i]
            bf += bfli * lidf[i]
            sob += sobli * lidf[i]
            sof += sofli * lidf[i]

        sdb, sdf = 0.5 * (ks + bf), 0.5 * (ks - bf)
        dob, dof = 0.5 * (ko + bf), 0.5 * (ko - bf)
        ddb, ddf = 0.5 * (1. + bf), 0.5 * (1. - bf)

        sigb = ddb * rho + ddf * tau
        sigf = ddf * rho + ddb * tau
        att = 1. - sigf
        m2 = np.maximum((att + sigb) * (att - sigb), 0)
        m = np.sqrt(m2)
        sb, sf = sdb * rho + sdf * tau, sdf * rho + sdb * tau
        vb, vf = dob * rho + dof * tau, dof * rho + dob * tau
        w = sob * rho + sof * tau

        e1, e2 = np.exp(-m * lai), np.exp(-2 * m * lai)
        rinf = (att - m) / sigb
        rinf2 = rinf * rinf
        re = rinf * e1
        denom = 1. - rinf2 * e2

        J1ks, J2ks = self._Jfunc1(ks, m, lai), self._Jfunc2(ks, m, lai)
        J1ko, J2ko = self._Jfunc1(ko, m, lai), self._Jfunc2(ko, m, lai)

        Ps, Qs = (sf + sb * rinf) * J1ks, (sf * rinf + sb) * J2ks
        Pv, Qv = (vf + vb * rinf) * J1ko, (vf * rinf + vb) * J2ko

        rdd = rinf * (1. - e2) / denom
        tdd = (1. - rinf2) * e1 / denom
        tsd = (Ps - re * Qs) / denom
        rsd = (Qs - re * Ps) / denom
        tdo = (Pv - re * Qv) / denom
        rdo = (Qv - re * Pv) / denom

        tss, too = np.exp(-ks * lai), np.exp(-ko * lai)
        z = self._Jfunc3(ks, ko, lai)
        g1, g2 = (z - J1ks * too) / (ko + m), (z - J1ko * tss) / (ks + m)

        Tv1, Tv2 = (vf * rinf + vb) * g1, (vf + vb * rinf) * g2
        T1, T2 = Tv1 * (sf + sb * rinf), Tv2 * (sf * rinf + sb)
        T3 = (rdo * Qs + tdo * Ps) * rinf
        rsod = (T1 + T2 - T3) / (1. - rinf2)

        alf = 1e6
        if q > 0.: alf = (dso / q) * 2. / (ks + ko)
        if alf > 200.: alf = 200.
        if alf == 0.:
            tsstoo = tss
            sumint = (1 - tss) / (ks * lai)
        else:
            fhot = lai * np.sqrt(ko * ks)
            x1, y1, f1 = 0., 0., 1.
            fint = (1. - np.exp(-alf)) * .05
            sumint = 0.
            for i in range(20):
                x2 = -np.log(1. - (i + 1) * fint) / alf if i < 19 else 1.
                y2 = -(ko + ks) * lai * x2 + fhot * (1. - np.exp(-alf * x2)) / alf
                f2 = np.exp(y2)
                sumint += (f2 - f1) * (x2 - x1) / (y2 - y1)
                x1, y1, f1 = x2, y2, f2
            tsstoo = f1

        rsos = w * lai * sumint
        dn = 1. - rsoil * rdd
        rddt = rdd + tdd * rsoil * tdd / dn
        rsdt = rsd + (tsd + tss) * rsoil * tdd / dn
        rdot = rdo + tdd * rsoil * (tdo + too) / dn

        rsot = rsos + tsstoo * rsoil + rsod + ((tss + tsd) * tdo + (tsd + tss * rsoil * rdd) * too) * rsoil / dn
        return rsot, rdot, rsdt, rddt

    def run(self, N, Cab, Car, Cbrown, Cw, Cm, LAI, hspot, tts, tto, psi, ala, Bs, soil_idx):
        lidf = self._campbell(13, ala)
        rho, tau = self._prospect_5B(N, Cab, Car, Cbrown, Cw, Cm)
        base_soil = self.custom_soils[soil_idx]
        rsoil0 = np.clip(base_soil * Bs, 0.0, 1.0)

        rsot, rdot, rsdt, rddt = self._PRO4SAIL(rho, tau, lidf, LAI, hspot, tts, tto, psi, rsoil0)
        return rsot, rdot


# 实例化模型
prosail_model = ProsailModel(spec_csv_path=PROSAIL_CSV_FILE, custom_soil_paths=CUSTOM_SOIL_FILES)


# ======================= 4. 核心算法：Roujean 核函数计算 =======================
def roujean_k_vol(sza, vza, phi):
    sza_r, vza_r, phi_r = np.radians(sza), np.radians(vza), np.radians(phi)
    cos_xi = np.cos(sza_r) * np.cos(vza_r) + np.sin(sza_r) * np.sin(vza_r) * np.cos(phi_r)
    phase = np.arccos(np.clip(cos_xi, -1.0, 1.0))
    term = (np.pi / 2.0 - phase) * np.cos(phase) + np.sin(phase)
    return (4.0 / (3.0 * np.pi)) * (term / (np.cos(sza_r) + np.cos(vza_r))) - (1.0 / 3.0)


def roujean_k_geo(sza, vza, phi):
    sza_r, vza_r, phi_r = np.radians(sza), np.radians(vza), np.radians(phi)
    tan_s, tan_v = np.tan(sza_r), np.tan(vza_r)
    delta = np.sqrt(np.maximum(0, tan_s ** 2 + tan_v ** 2 - 2.0 * tan_s * tan_v * np.cos(phi_r)))
    term1 = (np.pi - phi_r) * np.cos(phi_r) + np.sin(phi_r)
    return (1.0 / (2.0 * np.pi)) * term1 * tan_s * tan_v - (1.0 / np.pi) * (tan_s + tan_v + delta)


def integrate_diffuse_kernel_value():
    """精确半球双重积分"""
    sza_range = np.linspace(0, 89, 90)
    phi_range = np.linspace(0, 359, 36)
    k_vol_sum, k_geo_sum, weight_sum = 0, 0, 0
    for sza in sza_range:
        weight_sza = np.sin(np.radians(sza)) * np.cos(np.radians(sza))
        for phi in phi_range:
            k_vol_sum += roujean_k_vol(sza, VZA_FIXED, phi) * weight_sza
            k_geo_sum += roujean_k_geo(sza, VZA_FIXED, phi) * weight_sza
            weight_sum += weight_sza
    return k_vol_sum / weight_sum, k_geo_sum / weight_sum


K_VOL_DIFF, K_GEO_DIFF = integrate_diffuse_kernel_value()


def calculate_dlc_kernel(sza, vza, phi, f_lambda):
    k_vol_dir = roujean_k_vol(sza, vza, phi)
    k_geo_dir = roujean_k_geo(sza, vza, phi)
    k_vol_dlc = (1 - f_lambda) * k_vol_dir + f_lambda * K_VOL_DIFF
    k_geo_dlc = (1 - f_lambda) * k_geo_dir + f_lambda * K_GEO_DIFF
    return k_vol_dlc, k_geo_dlc


def add_gaussian_white_noise(refl_array, mult_noise_std=0.02, add_noise_std=0.01):
    refl_array = np.array(refl_array)
    mult_noise = np.random.normal(0, mult_noise_std, size=refl_array.shape)
    add_noise = np.random.normal(0, add_noise_std, size=refl_array.shape)
    return np.clip(refl_array * (1 + mult_noise) + add_noise, 0.0, 1.0)


# ======================= 5. LHS 数据采样与动态边界 =======================
def load_geometry_groups(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).lower() for c in df.columns]
    if 'doy' in df.columns:
        return [g for _, g in df.groupby('doy') if len(g) >= 3]
    return [df.iloc[i:i + 56] for i in range(0, len(df), 56)]


def get_weiss_parameters(n_samples):
    print("🎲 正在使用拉丁超立方 (LHS) 进行高密度参数空间采样 (已适配 Visionpoint 特征)...")
    sampler = qmc.LatinHypercube(d=8, seed=42)
    lhs_samples = sampler.random(n=n_samples)

    u_lai, u_cab, u_cw, u_hspot = lhs_samples[:, 0], lhs_samples[:, 1], lhs_samples[:, 2], lhs_samples[:, 3]
    u_ala, u_n, u_bs, u_soil = lhs_samples[:, 4], lhs_samples[:, 5], lhs_samples[:, 6], lhs_samples[:, 7]

    soil_idx = np.zeros(n_samples, dtype=int)
    soil_idx[(u_soil >= 0.50) & (u_soil < 0.85)] = 1
    soil_idx[u_soil >= 0.85] = 2

    min_u_lai_global, max_u_lai_global = np.exp(-0.5 * 7), np.exp(-0.5 * 0.1)
    mapped_u_lai = u_lai * (max_u_lai_global - min_u_lai_global) + min_u_lai_global
    lai = -2 * np.log(mapped_u_lai)

    mapped_cab = u_cab * (np.exp(-0.01 * 1.0) - np.exp(-0.01 * 65.0)) + np.exp(-0.01 * 65.0)
    cab = -100 * np.log(mapped_cab)

    cw = -1 / 50 * np.log(u_cw * (np.exp(-50 * 0.005) - np.exp(-50 * 0.025)) + np.exp(-50 * 0.025))
    hspot = -1 / 3 * np.log(u_hspot * (np.exp(-3 * 0.05) - np.exp(-3 * 1)) + np.exp(-3 * 1))
    ala = np.rad2deg(np.arccos(u_ala * (np.cos(np.deg2rad(20)) - np.cos(np.deg2rad(75))) + np.cos(np.deg2rad(75))))
    n_struct = u_n * (2.5 - 1.0) + 1.0
    bs = u_bs * (1.6 - 0.7) + 0.5

    lai = np.clip(lai, 0.1, 7.0)
    cab = np.clip(cab, 1.0, 65.0)

    # 🌟 注入 3% 的纯背景锚点数据 (LAI=0, Cab=0) 以增强低值段稳定性
    print("🌱 正在注入纯背景边界锚点数据...")
    np.random.seed(42)
    pure_bg_mask = np.random.rand(n_samples) < 0.03
    lai[pure_bg_mask] = 0.0
    cab[pure_bg_mask] = 0.0
    cw[pure_bg_mask] = 0.0

    return pd.DataFrame({
        "lai": lai, "cab": cab, "cw": cw, "hspot": hspot,
        "ala": ala, "n": n_struct, "bs": bs, "soil_idx": soil_idx
    })


# ======================= 6. 核心工作流：27参数联合求解 =======================
def process_case(args):
    case_id, bio_dict, geo_list = args
    curr_cm, curr_bs, curr_soil_idx = bio_dict['cw'] / 4.0, bio_dict['bs'], int(bio_dict['soil_idx'])

    time_series_data = []

    for row in geo_list:
        tts = row.get('cos_theta_s', row.get('tts', 0.0))
        if tts <= 1.0: tts = np.degrees(np.arccos(np.clip(tts, -1.0, 1.0)))
        if tts > 60: continue

        saa = row.get('sun_azimuth', row.get('saa', 180.0))
        psi = np.abs(saa - SENSOR_AZIMUTH)
        if psi > 180: psi = 360 - psi
        f_par = row.get('f_par', 0.2)

        rsot_spec, rdot_spec = prosail_model.run(
            N=bio_dict['n'], Cab=bio_dict['cab'], Car=8, Cbrown=0, Cw=bio_dict['cw'], Cm=curr_cm,
            LAI=bio_dict['lai'], hspot=bio_dict['hspot'], tts=tts, tto=VZA_FIXED,
            psi=psi, ala=bio_dict['ala'], Bs=curr_bs, soil_idx=curr_soil_idx
        )

        rho_dir = rsot_spec[BAND_INDICES]
        rho_hdr = rdot_spec[BAND_INDICES]

        moment_refls_clean = []
        f_lams_list = []

        for i, band in enumerate(BAND_NAMES):
            # 动态使用 6S 系数计算各波段散射分数
            f_lam = np.clip(np.poly1d(F_LAMBDA_COEFFS[band])(f_par) if f_par <= 0.9 else f_par, 0, 1)
            f_lams_list.append(f_lam)
            refl = (1 - f_lam) * rho_dir[i] + f_lam * rho_hdr[i]
            moment_refls_clean.append(refl)

        noisy_refls = add_gaussian_white_noise(moment_refls_clean)
        time_series_data.append({'sza': tts, 'phi': psi, 'f_lams': f_lams_list, 'abs_refls_noisy': noisy_refls})

    T = len(time_series_data)
    if T < 5: return None

    X_sza = np.array([m['sza'] for m in time_series_data])
    X_vza = np.full(T, VZA_FIXED)
    X_phi = np.array([m['phi'] for m in time_series_data])

    obs_abs_refls = np.zeros((9, T))
    f_lams_matrix = np.zeros((9, T))
    for i in range(9):
        obs_abs_refls[i, :] = [m['abs_refls_noisy'][i] for m in time_series_data]
        f_lams_matrix[i, :] = [m['f_lams'][i] for m in time_series_data]

    # 计算目标相对反射率
    spectral_means = np.mean(obs_abs_refls, axis=0)
    y_rel_matrix = obs_abs_refls / (spectral_means + 1e-6)

    k_vol_dlc_all = np.zeros((9, T))
    k_geo_dlc_all = np.zeros((9, T))
    for i in range(9):
        k_vol_dlc_all[i], k_geo_dlc_all[i] = calculate_dlc_kernel(X_sza, X_vza, X_phi, f_lams_matrix[i])

    # 🚀 27 参数全局联合优化 (9波段 x 3参数)
    def global_cost_func(params):
        params_2d = params.reshape(9, 3)
        X_mod_abs = np.zeros((9, T))

        for i in range(9):
            k0, k1, k2 = params_2d[i]
            X_mod_abs[i] = k0 + k1 * k_vol_dlc_all[i] + k2 * k_geo_dlc_all[i]

        mean_X_mod = np.mean(X_mod_abs, axis=0)
        X_mod_rel = X_mod_abs / (mean_X_mod + 1e-6)

        # 光谱形状拟合误差
        shape_error = np.sum((y_rel_matrix - X_mod_rel) ** 2)
        # 尺度锚定：防止相对反演时分母尺度坍塌
        scale_penalty = 100.0 * np.sum((mean_X_mod - 1.0) ** 2)

        return shape_error + scale_penalty

    # 初始值设定
    x0 = []
    for i in range(9):
        x0.extend([np.mean(y_rel_matrix[i, :]), 0.05, 0.05])

    bounds_27 = [(0, None), (-0.05, None), (-0.05, None)] * 9

    res = minimize(global_cost_func, x0=x0, method='SLSQP', bounds=bounds_27)

    kernel_coeffs = {}
    if res.success:
        params_opt = res.x.reshape(9, 3)
        for i, band in enumerate(BAND_NAMES):
            kernel_coeffs[f"{band}_k0"] = params_opt[i, 0]
            kernel_coeffs[f"{band}_k1"] = params_opt[i, 1]
            kernel_coeffs[f"{band}_k2"] = params_opt[i, 2]
    else:
        return None

    result_row = {'case_id': case_id, 'lai': bio_dict['lai'], 'cab': bio_dict['cab'],
                  'ccc': bio_dict['lai'] * bio_dict['cab'],
                  'cw': bio_dict['cw'], 'ala': bio_dict['ala'], 'bs': bio_dict['bs'], 'soil_idx': bio_dict['soil_idx']}
    result_row.update(kernel_coeffs)
    return result_row


# ======================= 7. 主程序启动 =======================
if __name__ == "__main__":
    print(f"🚀 初始化 PROSAIL-Kernel 联合反演数据集生成引擎 (N={N_CASES})...")

    daily_geometries = load_geometry_groups(GEOMETRY_FILE)
    df_bio = get_weiss_parameters(N_CASES)

    print("⏳ 正在组装多进程任务队列...")
    tasks = []
    for i in range(N_CASES):
        bio_dict = df_bio.iloc[i].to_dict()
        geo_df = daily_geometries[np.random.randint(0, len(daily_geometries))]
        geo_list = geo_df.to_dict('records')
        tasks.append((i, bio_dict, geo_list))

    print(f"⚡ 启动多进程计算 (27参数联合优化 + 尺度锚定)...")
    with Pool(cpu_count()) as p:
        results = [r for r in tqdm(p.imap(process_case, tasks), total=N_CASES) if r is not None]

    if results:
        final_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        final_df.to_csv(SAVE_PATH, index=False)
        print(f"✅ 完成! 样本已保存至: {SAVE_PATH}")
        print(f"🔥 数据集已准备就绪，可以开始构建神经网络反演模型了！")
    else:
        print("❌ 严重错误：未产生任何有效模拟案例。")