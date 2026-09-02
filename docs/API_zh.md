# PyTICC 接口说明

本文档说明 PyTICC 0.1.x 当前可供用户调用的接口。内容以 `pyticc.__all__`、高层散射流程和现有示例为准；未在本文列出的 `pyticc.*` 子模块函数主要是内部数值实现，不保证接口稳定。

本文既是入门说明，也是参数参考。第一次使用建议依次阅读第 1、2、5、6、7、8 节；接入新势能面时重点阅读第 3 节；输入文件用户重点阅读第 11 节；开壳层精细结构计算重点阅读第 5 节。

## 目录

- [1. 基本约定](#1-基本约定)
- [2. 两种调用入口](#2-两种调用入口)
- [3. PES 接口](#3-pes-接口)
- [4. 单体和基组接口](#4-单体和基组接口)
- [5. 精细结构接口](#5-精细结构接口)
- [6. 体系与通道](#6-体系与通道)
- [7. Hamiltonian 构造](#7-hamiltonian-构造)
- [8. 传播与求解](#8-传播与求解)
- [9. 结果对象](#9-结果对象)
- [10. 文本报告接口](#10-文本报告接口)
- [11. TOML 输入结构](#11-toml-输入结构)
- [12. 单位、质量与辅助函数](#12-单位质量与辅助函数)
- [13. 错误处理和资源管理](#13-错误处理和资源管理)
- [14. 公开 API 速查](#14-公开-api-速查)

## 1. 基本约定

PyTICC 默认使用原子单位：

| 物理量 | 输入/输出单位 |
| --- | --- |
| 长度、径向传播坐标 | bohr |
| 能量、势能、通道阈值 | Hartree |
| 质量 | 电子质量 |
| 角度 | radian |
| 电场强度 | 原子单位 |
| 磁自旋偶极系数 $C_{\rm dd}$ | Hartree·bohr³ |

精细结构量子数允许半整数。为避免浮点数作为离散标签，公开接口统一用两倍量子数表示：

- `two_j = 2j`
- `two_J = 2J`
- `two_K = 2K`
- `two_S = 2S`
- `two_lambda_abs = 2|Λ|`

例如，$j=3/2$ 写为 `two_j=3`，$S=1/2$ 写为 `two_S=1`，$|Λ|=1$ 写为 `two_lambda_abs=2`。

常用换算常数可直接从 `pyticc` 导入：

```python
import pyticc as ticc

energy_au = 100.0 * ticc.CM2AU
energy_cm = energy_au * ticc.AU2CM
length_au = 1.0 * ticc.ANG2AU
length_angstrom = length_au * ticc.AU2ANG

# 单个光谱常数也可按单位名转换
spin_orbit_au = ticc.energy_to_au(-139.2, "cm-1")
frequency_au = ticc.energy_to_au(1667.0, "MHz")
```

`energy_to_au` 接受 `"au"`、`"cm-1"`、`"Hz"`、`"kHz"`、`"MHz"` 和 `"GHz"`。对应的双向常数包括 `CM2AU/AU2CM`、`HZ2AU/AU2HZ`、`KHZ2AU/AU2KHZ`、`MHZ2AU/AU2MHZ` 和 `GHZ2AU/AU2GHZ`。

数组索引和形状遵循以下约定：

- 电子态编号从 `0` 开始。
- 振动、转动和螺旋度量子数分别记为 `v`、`j` 和 `K`。
- 总能量数组形状为 `(n_energy,)`。
- LogD 矩阵批次形状为 `(n_energy, n_channel, n_channel)`。
- 每个能量的开放通道数可能不同，因此 S 矩阵使用 `tuple` 保存；`Smat[i]` 的形状为 `(n_open[i], n_open[i])`。
- 势能回调函数必须返回有限值，并严格满足本文给出的数组形状；除总自旋分辨 2+2 orbital PES 可返回复厄米矩阵外，其余当前接口返回实数。
- 通道在 `E_int < Etot` 时视为开放；阈值恰好等于总能量时仍归为关闭通道。
- S 矩阵使用 `Smat[outgoing, incoming]` 的行列约定。

## 2. 两种调用入口

### 2.1 TOML 高层入口

```python
import pyticc as ticc

result = ticc.run("input.toml")
```

函数签名：

```python
run(
    source: str | Path,
    *,
    pes: PESWrapper | DiabaticPESWrapper | None = None,
) -> ScatteringResult | CoupledStatesResult
```

- `source`：TOML 输入文件路径。文件内的相对路径均相对于该文件所在目录解析。
- `pes`：可选的已构造 PES 对象。省略时，根据 TOML 的 `[pes]` 段编译或加载 Fortran PES。
- 当前 TOML 入口支持 `A+BC`、`A+BC_electric`、`A+BC_diabatic` 和 `AB+CD`。
- 原子–三原子和 Delves 反应散射目前使用 Python 接口。

### 2.2 Python 组合式入口

推荐的 Python 调用流程为：

1. 构造或加载 PES。
2. 准备单体基组。
3. 用 `build_ScattSystem` 构造物理体系和通道。
4. 用 `prepare_potential` 一次性建立径向/求积格点并计算原始 PES 值。
5. 用 `solve` 在内部构造 Hamiltonian、传播并匹配。

以绝热原子–双原子散射为例：

```python
from pathlib import Path

import numpy as np

import pyticc as ticc

pes_dir = Path("pes")
pes = ticc.load_fortran_pes(
    [pes_dir / "interaction-PES.f"],
    pes_dir / "pyticc_wrapper.f90",
    workdir=pes_dir,
)

try:
    mass_ar, mass_h, mass_f = ticc.element_masses_au("Ar", "H", "F")
    mass_hf = mass_h + mass_f

    hf = ticc.prepare_Diatom(
        pes.monomer_Y,
        r=(1.5, 4.5),
        n_dvr=100,
        n_podvr=5,
        vmax=0,
        jmax=4,
        mass=ticc.reduced_mass(mass_h, mass_f),
    )
    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        hf,
        scattering_type="A+BC",
        Jtot=0,
        system_parity=1,
        channel=ticc.ChannelSpec(E_Y_cut=2000.0 * ticc.CM2AU),
        potential=pes,
        reduced_mass=ticc.reduced_mass(mass_ar, mass_hf),
    )
    potential_grid = ticc.prepare_potential(
        system,
        (4.5, 6.5, 8.0, 12.0),
        (0.05, 0.08, 0.10),
        n_theta=35,
        processes=4,
    )
    result = ticc.solve(
        system,
        np.array([100.0, 300.0, 500.0]) * ticc.CM2AU,
        potential_grid,
        ticc.Propagation(),
    )
finally:
    pes.close()
```

## 3. PES 接口

### 3.1 `PESWrapper`

```python
PESWrapper(
    interaction,
    monomer_X=None,
    monomer_Y=None,
    interaction_many=None,
)
```

绝热、非反应散射使用该对象。

| 回调 | 输入 | 返回 |
| --- | --- | --- |
| `interaction(R, coordinates)` | `R: float`；`coordinates: (n_coordinate, n_grid)` | 相互作用能 `(n_grid,)` |
| `monomer_X(r)` | 键长 `(n_grid,)` | X 单体势能 `(n_grid,)` |
| `monomer_Y(r)` | 键长 `(n_grid,)` | Y 单体势能 `(n_grid,)` |
| `interaction_many(R, coordinates)` | `R: (n_R,)`；坐标 `(n_coordinate, n_grid)` | `(n_R, n_grid)` |

`interaction_many` 是可选的批量径向接口；提供它可以减少 Python 调用和 PES 重复初始化的开销。相互作用势应只返回单体之间的相互作用能，不应重复加入单体势能。

不同几何下 `coordinates` 的行顺序为：

| 几何 | 坐标行顺序 |
| --- | --- |
| 原子–双原子 | `(r, theta)` |
| 原子–三原子 | `(r_1, r_2, theta_1, theta_2, phi)` |
| 双原子–双原子 | `(r_X, r_Y, theta_X, theta_Y, phi)` |

### 3.2 `DiabaticPESWrapper`

```python
DiabaticPESWrapper(
    n_state: int,
    monomer,
    interaction,
    interaction_many=None,
)
```

用于非绝热原子–双原子散射：

- `monomer(r)`：输入 `(n_r,)`，返回 `(n_r, n_state)`。
- `interaction(R, coordinates)`：输入坐标行 `(r, theta)`，返回 `(n_grid, n_state, n_state)`。
- `interaction_many(R, coordinates)`：返回 `(n_R, n_grid, n_state, n_state)`。
- 相互作用矩阵必须是实对称矩阵；对角元应已减去对应电子态的单体势。
- `monomer_values(r)` 返回全部电子态的单体势；`monomer_state(i)` 返回第 `i` 个电子态的单体势回调。

### 3.3 `LambdaPES`

```python
LambdaPES(
    interaction,
    monomer_Y=None,
    interaction_many=None,
)
```

`LambdaPES` 用于一个固定 $|Λ|$ 电子流形内的开壳层原子–双原子精细结构散射。它在有符号电子基 $\{+Λ,-Λ\}$ 中提供两个势能分量：

$$
V_{\mathrm{sum}}=\frac{V_{A''}+V_{A'}}{2},\qquad
V_{\mathrm{dif}}=\frac{V_{A''}-V_{A'}}{2}.
$$

`V_sum` 是有符号 Λ 基中的对角元，`V_dif` 是翻转 Λ 的耦合元。注意这里采用的差值顺序是 `A'' - A'`，不能与相反符号的 PES 约定混用。

| 回调 | 输入 | 返回 |
| --- | --- | --- |
| `interaction(R, coordinates)` | `R: float`；原子–双原子坐标 `(2, n_grid)`，行顺序 `(r, theta)` | `(n_grid, 2)`，末轴依次为 `(V_sum, V_dif)` |
| `monomer_Y(r)` | 双原子键长 `(n_r,)` | 孤立双原子势 `(n_r,)` |
| `interaction_many(R, coordinates)` | `R: (n_R,)`；坐标 `(2, n_grid)` | `(n_R, n_grid, 2)` |

对 $Λ=0$ 的 $Σ$ 态，可把普通标量 PES 适配为 Λ 接口：

```python
lambda_pes = ticc.as_lambda_pes(scalar_pes)
```

适配后 `V_sum` 等于原标量势，`V_dif` 恒为零；`monomer_Y` 和批量径向接口会一并保留。`lambda_pes.close()` 会转发给原 `PESWrapper.close()`，因此不要把两个对象当作彼此独立的资源重复管理。

### 3.4 `SpinResolvedDiatomDiatomPES`

```python
SpinResolvedDiatomDiatomPES(
    interaction,
    two_total_spins,
    orbital_states,
    interaction_many=None,
)
```

该接口用于两个精细结构双原子的总电子自旋分辨相互作用。电子势写成

$$
\hat V_{\rm el}(Q)
=\sum_{\mathcal S}\hat P_{\mathcal S}\otimes
\hat W^{(\mathcal S)}_{\rm orb}(Q),
$$

其中 `two_total_spins[s] = 2𝒮`，`orbital_states[alpha]` 给出有符号
$(2\Lambda_X,2\Lambda_Y)$ 乘积态。`interaction` 的坐标行顺序为
`(r_X, r_Y, theta_X, theta_Y, phi)`。

| 回调 | 输入 | 返回 |
| --- | --- | --- |
| `interaction(R, coordinates)` | `R: float`；坐标 `(5,n_grid)` | `(n_grid,n_spin,n_orbital,n_orbital)` |
| `interaction_many(R, coordinates)` | `R: (n_R,)`；坐标 `(5,n_grid)` | `(n_R,n_grid,n_spin,n_orbital,n_orbital)` |

每个 orbital matrix 必须是有限的复厄米矩阵 $W=W^\dagger$，单位 Hartree；实对称矩阵是完全支持的特例。两个电子轴都按
`orbital_states` 排列；构造散射体系时，该元组必须完整覆盖两个单体允许的所有有符号
$\Lambda$ 乘积态。例如两个 $^2\Pi$ 单体的完整有符号轨道乘积基可写成：

```python
orbitals = (
    ticc.OrbitalState(-2, -2),
    ticc.OrbitalState(-2, 2),
    ticc.OrbitalState(2, -2),
    ticc.OrbitalState(2, 2),
)
pes = ticc.SpinResolvedDiatomDiatomPES(
    interaction,
    two_total_spins=(0, 2),  # singlet/triplet: 2𝒮 = 0, 2
    orbital_states=orbitals,
)
```

若普通标量 `PESWrapper` 对所有总自旋面相同且不翻转 $\Lambda$，可使用：

```python
pes = ticc.as_spin_resolved_diatom_diatom_pes(
    scalar_pes,
    two_S_X=1,
    two_lambda_X_abs=2,
    two_S_Y=1,
    two_lambda_Y_abs=2,
)
```

适配器在所有允许的 $\mathcal S$ 上复制标量势，并在 orbital space 中乘单位矩阵。

### 3.5 `TotalPES`

```python
TotalPES(potential)
```

Delves 三原子反应散射使用总势能面。`potential(bonds)` 接收形状 `(3, n_grid)`、行顺序为 `(r_AB, r_BC, r_CA)` 的三个物理键长，返回 `(n_grid,)` 的总能量。

该接口要求势能面直接返回一个绝热电子面上的总能量，不得减去双原子势、额外加入单体势或自行改变能量零点。

### 3.6 Fortran PES 加载

```python
load_fortran_pes(sources, wrapper=None, *, workdir=None, lapack=False)
load_fortran_diabatic_pes(
    sources, wrapper=None, *, n_state=2, workdir=None, lapack=False
)
load_fortran_lambda_pes(sources, wrapper=None, *, workdir=None, lapack=False)
load_fortran_total_pes(sources, wrapper=None, *, workdir=None, lapack=False)
```

- `sources` 可为一个或多个 Fortran 源文件，也可为描述源文件和 wrapper 的短 TOML 文件。
- `workdir` 是 PES 运行时数据文件所在目录。
- `processes` 不属于加载器；它在 `prepare_potential(..., processes=N)` 中控制这一次 PES 格点计算。
- `lapack=True` 表示 PES 编译时需要链接 LAPACK。
- 加载结果持有编译模块资源时，应在使用结束后调用 `close()`；重复调用是安全的。

Fortran wrapper 的必需子程序为：

| 类型 | 必需子程序 | 可选子程序 |
| --- | --- | --- |
| 绝热相互作用 PES | `pyticc_interaction_grid` | `pyticc_monomer_x_grid`、`pyticc_monomer_y_grid` |
| 非绝热 PES | `pyticc_diabatic_interaction_grid`、`pyticc_diabatic_monomer_grid` | 无 |
| Λ 分辨 PES | `pyticc_lambda_grid` | `pyticc_monomer_y_grid` |
| 三原子总 PES | `pyticc_total_grid` | 无 |

标量相互作用 wrapper 的核心 ABI 为：

```fortran
subroutine pyticc_interaction_grid(RR, coordinates, V, n_coordinate, n_grid)
    real*8, intent(in) :: RR
    real*8, intent(in) :: coordinates(n_coordinate, n_grid)
    real*8, intent(out) :: V(n_grid)
end subroutine pyticc_interaction_grid

subroutine pyticc_monomer_y_grid(r, V, n_grid)
    real*8, intent(in) :: r(n_grid)
    real*8, intent(out) :: V(n_grid)
end subroutine pyticc_monomer_y_grid
```

非绝热 wrapper 的核心 ABI 为：

```fortran
subroutine pyticc_diabatic_interaction_grid(RR, coordinates, V, n_coordinate, n_grid, n_state)
    real*8, intent(in) :: RR
    real*8, intent(in) :: coordinates(n_coordinate, n_grid)
    real*8, intent(out) :: V(n_grid, n_state, n_state)
    integer, intent(in) :: n_state
end subroutine pyticc_diabatic_interaction_grid

subroutine pyticc_diabatic_monomer_grid(r, V, n_grid, n_state)
    real*8, intent(in) :: r(n_grid)
    real*8, intent(out) :: V(n_grid, n_state)
    integer, intent(in) :: n_state
end subroutine pyticc_diabatic_monomer_grid
```

Λ 分辨 wrapper 的核心 ABI 为：

```fortran
subroutine pyticc_lambda_grid(RR, coordinates, V, n_coordinate, n_grid)
    real*8, intent(in) :: RR
    real*8, intent(in) :: coordinates(n_coordinate, n_grid)
    real*8, intent(out) :: V(n_grid, 2)
end subroutine pyticc_lambda_grid
```

三原子总势 wrapper 的核心 ABI 为：

```fortran
subroutine pyticc_total_grid(bonds, V, n_grid)
    real*8, intent(in) :: bonds(3, n_grid)
    real*8, intent(out) :: V(n_grid)
end subroutine pyticc_total_grid
```

`n_grid`、`n_coordinate` 在生成的 f2py 签名中由数组形状推导并隐藏；Fortran 源中仍应保留这些形参。所有数组均应按上述维度声明，wrapper 负责把 PyTICC 的 bohr/radian/Hartree 契约转换为原生 PES 的原子顺序、坐标与能量单位。

具体实现可参考 `example/ArHF/pes/pyticc_wrapper.f90`、`example/HO2_diabatic/pes/pyticc_wrapper.f90`、`example/HeOH_2Pi/pes/pyticc_lambda_wrapper.f90` 和 `example/H3_Delves/pyticc_total_wrapper.f90`。

### 3.7 Fortran PES 短 TOML

当 `wrapper=None` 且 `sources` 是单个 TOML 路径时，四个 Fortran 加载函数都可从短配置读取源文件：

```toml
sources = ["native_pes.f", "support.f90"]
wrapper = "pyticc_lambda_wrapper.f90"
workdir = "."
lapack = false
```

其中 `sources`、`wrapper` 和 `workdir` 相对于该短 TOML 文件解析。Python 调用中显式传入的 `workdir` 优先于 TOML 的同名值；`lapack=True` 与 TOML 中的 `lapack=true` 取逻辑“或”。

## 4. 单体和基组接口

### 4.1 无内部结构原子

```python
atom = ticc.AtomSpec()
```

`AtomSpec` 表示没有内部自由度、内能为零的原子。

### 4.2 绝热双原子

推荐直接使用：

```python
prepare_Diatom(
    potential,
    *,
    r: tuple[float, float],
    n_dvr: int,
    n_podvr: int,
    vmax: int,
    jmax: int,
    mass: float,
    energy_zero: float | None = None,
) -> DiatomBasis
```

- `r`：Sine DVR 左右边界。
- `mass`：双原子约化质量。
- `energy_zero=None`：默认使用 `E(v=0,j=0)` 作为通道能量零点。
- 返回对象的 `Eint[v, j]` 为相对 `energy_zero` 的转振能级。

`DiatomBasis` 的主要数据为：

| 成员 | 含义与形状 |
| --- | --- |
| `rovib.grids` | PODVR 径向网格，`(n_podvr,)` |
| `rovib.E_vj` | 绝对转振能量，索引 `[v, j]`，`(vmax+1, jmax+1)` |
| `rovib.WF_vj` | 径向波函数，索引 `[grid, v, j]`，`(n_podvr, vmax+1, jmax+1)` |
| `energy_zero` | 从散射通道阈值中减去的绝对能量 |
| `Eint` | `rovib.E_vj - energy_zero` |

需要复用原始 DVR 时，可先调用：

```python
build_SineDVR(
    a: float,
    b: float,
    n_dvr: int,
    mass: float,
    pot_func,
) -> SineDVR

build_DiatomBasis(
    dvr,
    *,
    n_podvr: int,
    vmax: int,
    jmax: int,
    mass: float,
    energy_zero: float | None = None,
) -> DiatomBasis
```

`pot_func` 必须向量化地接收 `(n_dvr,)` 的键长并返回 `(n_dvr,)` 的势能。`SineDVR` 保存 `grids`、标量积分权重 `weights`、`dvr_to_fbr`、本征值 `eigen_val` 和按列存放的本征矢 `eigen_vec`。

### 4.3 多电子态双原子

```python
prepare_DiabaticDiatom(
    potential,
    *,
    n_state: int,
    r: tuple[float, float],
    n_dvr: int,
    n_podvr: int | Sequence[int],
    vmax: int | Sequence[int],
    jmax: int | Sequence[int],
    mass: float,
    energy_zero: float | None = None,
) -> DiabaticDiatomBasis
```

`n_podvr`、`vmax` 和 `jmax` 可以是所有电子态共享的整数，也可以是按电子态排列的序列。所有电子态共享同一套 primitive DVR 网格和同一个能量零点。

`DiabaticDiatomBasis.states[i]` 对应从零开始的电子态 `i`，每个状态同时保存：

- `contracted`：用于 PES 投影的 PODVR `RovibBasis`。
- `primitive`：公共 primitive DVR 网格上的 `RovibBasis`。
- `vmax`、`jmax`：该电子态实际保留的最大量子数。

若 `energy_zero=None`，使用第 0 个电子态的 `E(v=0,j=0)`。传入逐电子态序列时，序列长度必须严格等于 `n_state`。

### 4.4 电场修饰双原子

```python
prepare_DiatomElectric(
    potential,
    response,
    *,
    r,
    n_dvr,
    n_podvr,
    electric_strength,
    jmax,
    M,
    lmax,
    n_alpha,
    mass,
    energy_zero=None,
) -> DiatomElectricBasis
```

- `response` 为 `ElectricResponseTable` 或固定格式 CSV 路径。
- CSV 由 `load_electric_response_csv(path)` 读取，列为 `r, mu_z, alpha_xx, alpha_zz, beta_zzz, beta_xxz`。
- `M` 是总体系在空间固定电场轴上的守恒投影。
- `n_alpha` 是每个固定 `m` 分块保留的最低修饰态数目。
- `energy_zero=None` 时使用 `m=0` 分块的最低本征值。

响应 CSV 的首行必须严格为：

```text
r,mu_z,alpha_xx,alpha_zz,beta_zzz,beta_xxz
```

所有列均使用原子单位，至少需要两行且 `r` 不能重复。读取后会按 `r` 排序；`ElectricResponseTable.evaluate(r)` 使用自然三次样条插值，并拒绝超出表格键长范围的外推。

`DiatomElectricBasis.blocks` 按 `m` 升序保存固定投影分块。每个分块中的 `energies` 形状为 `(n_alpha,)`，`coefficients[p, j_index, alpha]` 形状为 `(n_podvr, n_j, n_alpha)`。可用 `basis.block(m)` 取分块，用 `basis.relative_energies(m)` 取得相对公共零点的能量。

### 4.5 原子–三原子和 Delves 基组

原子–三原子单体基组推荐使用：

```python
prepare_Triatom(
    potential,
    *,
    r: tuple[tuple[float, float], tuple[float, float]],
    n_dvr: tuple[int, int],
    n_podvr: tuple[int, int],
    vmax: tuple[int, int],
    masses: tuple[float, float, float],
    equilibrium: tuple[float, float, float],
    n_theta: int,
    j1max: int,
    j2max: int,
    tmax: int,
    parity_block_sign: int,
    exchange_parity: int = 0,
    energy_zero: float | None = None,
    matching_tolerance: float = 1e-8,
    Kmax: int | None = None,
) -> TriatomBasis
```

- `potential` 接收 `(3, n_point)` 的三原子内坐标并返回 `(n_point,)`。
- `r` 和 `n_dvr` 分别给出两个径向 sine-DVR 的区间和格点数。用户只传入一个物理三原子 PES；`prepare_Triatom` 在 `equilibrium` 固定另两个坐标，内部自动生成两条一维参考势，不需要定义 `reference_1`/`reference_2` 回调。
- `masses` 的顺序是原子 `(A, B, C)`；`equilibrium` 的顺序是 `(r1, r2, theta1)`。
- `j1max` 是弯曲角动量截断，`j2max` 是三原子总转动角动量截断，`tmax` 是保留的收缩态最大编号。
- `parity_block_sign` 为 K=0 基中的 $\epsilon(-1)^J$，只能取 `-1` 或 `1`。
- `exchange_parity` 对 ABC 体系取 `0`；A2B 对称体系取 `-1` 或 `1`，且两个径向基大小必须一致。
- `Kmax=0` 只构造 K=0 收缩块，适合 `J=0`；省略时保留正 K 收缩块。
- 返回的 `Eint[j2, t]`、`K0_available[j2, t]` 和 `positive_K_available[j2, t]` 形状均为 `(j2max+1, tmax+1)`。

`build_TriatomBasis` 是保留的低层接口，供已经自行构造两个 `SineDVR` 对象的代码使用。

三原子 Delves 反应散射先准备物理信息：

```python
prepare_Delves(
    total_potential: TotalPES,
    mass: tuple[float, float, float],
    *,
    energy_zero: str = "native",
    scaled_r_step: float = 0.01,
    scaled_r_scan_max: float = 10.0,
) -> DelvesMonomer
```

`energy_zero` 当前支持：

- `"native"`：沿用 PES 原生能量零点。
- `"minimum"`：减去扫描得到的最低渐近双原子势阱能量；结果中的 `energy_zero` 可用于恢复 PES 原生能量约定。

## 5. 精细结构接口

精细结构接口用于无内部结构原子与开壳层双原子的全维场自由散射。当前覆盖任意 $^{2S+1}\Sigma$、$^{2S+1}\Pi$ 和 $^{2S+1}\Delta$ 电子态，在有符号 Hund case (a) primitive basis 中构造自旋–轨道、分子转动、自旋–自旋、自旋–转动和 Λ-doubling 项，再对每个固定 `(v, j, epsilon)` 分块对角化。最终的 `tau` 本征态可处于 case (a)、case (b) 或中间耦合极限。

当前散射层只支持 exact CC，且相互作用势假定与电子自旋无关。

### 5.1 `FSConstants`

```python
FSConstants(
    A=0.0,
    B=0.0,
    gamma=0.0,
    lambda_ss=0.0,
    O=0.0,
    P=0.0,
    Q=0.0,
    M=0.0,
    N=0.0,
)
```

所有字段在对象内部都使用 Hartree：

| 字段 | 物理含义 |
| --- | --- |
| `A` | 自旋–轨道常数 |
| `B` | 分子转动常数 |
| `gamma` | 自旋–转动常数 |
| `lambda_ss` | 自旋–自旋常数 |
| `O, P, Q` | Π/Δ 态 Λ-doubling 常数 |
| `M, N` | Δ 态的更高阶 Λ-doubling 常数 |

同一单位的一组常数可直接转换：

```python
constants = ticc.FSConstants.from_unit(
    "cm-1",
    A=-139.2,
    B=18.5487,
    gamma=-0.1342,
    P=0.2354,
    Q=-0.0057,
)
```

若一组常数混用不同单位，应分别调用 `energy_to_au` 后再使用普通构造器。

### 5.2 精细结构常数 CSV

```python
table = ticc.load_fs_constants_csv("constant_2Pi_OH.csv")
constants_v0 = table.for_v(0)
```

CSV 使用长表格式，表头必须严格为：

```csv
v,constant,value,unit
0,A,-139.2,cm-1
0,B,18.5487,cm-1
0,gamma,-0.1342,cm-1
0,P,0.2354,cm-1
0,Q,-0.0057,cm-1
```

详细规则：

- `v` 必须是非负十进制整数。
- `constant` 只能是 `A`、`B`、`gamma`、`lambda_ss`、`O`、`P`、`Q`、`M` 或 `N`。
- `unit` 只能是 `au`、`cm-1`、`Hz`、`kHz`、`MHz` 或 `GHz`。
- 同一 `(v, constant)` 不能重复；同一振动态中未出现的常数自动取零。
- 以 `#` 开头的整行会被忽略，可用于记录数据来源。
- `FSConstantsTable.vibrational_levels` 给出可用的 `v`；`for_v(v)` 在缺少该振动态时抛出 `ValueError`。

### 5.3 `prepare_fs_monomer`

```python
prepare_fs_monomer(
    potential,
    *,
    r: tuple[float, float],
    n_dvr: int,
    n_podvr: int,
    vmax: int,
    mass: float,
    two_j_values: Sequence[int],
    two_lambda_abs: int,
    two_S: int,
    constants: Sequence[FSConstants] | FSConstants | FSConstantsTable | str | Path,
    reflection_parity: int = 1,
    energy_zero: float | None = None,
) -> FSMonomerBasis
```

| 参数 | 说明 |
| --- | --- |
| `potential` | 孤立双原子标量势；输入 `(n_r,)` bohr，输出 `(n_r,)` Hartree |
| `r` | Sine DVR 键长区间，bohr |
| `n_dvr` | primitive Sine DVR 网格数 |
| `n_podvr` | 收缩 PODVR 网格数 |
| `vmax` | 最大振动量子数 |
| `mass` | 双原子约化质量 |
| `two_j_values` | 保留的 `2j`，例如 `(1, 3, 5)` 表示 `j=1/2,3/2,5/2` |
| `two_lambda_abs` | `2|Λ|`：Σ、Π、Δ 态分别为 `0`、`2`、`4` |
| `two_S` | `2S`；二重态、三重态分别为 `1`、`2` |
| `constants` | 所有 `v` 共用一组常数、每个 `v` 一组常数、已加载表，或 CSV 路径 |
| `reflection_parity` | Σ 电子态反射宇称，`Σ+` 为 `1`，`Σ-` 为 `-1`；Π/Δ 态通常保持默认值 |
| `energy_zero` | 散射阈值零点；`None` 使用所有分块中的最低本征能 |

`constants` 为序列时，其长度必须等于 `vmax+1`；为 CSV 或 `FSConstantsTable` 时必须覆盖 `v=0..vmax`。

返回的 `FSMonomerBasis` 保存：

| 成员 | 含义 |
| --- | --- |
| `vib.grids` | 共享 PODVR 径向网格，`(n_podvr,)` |
| `vib.energies` | 振动参考能，`(vmax+1,)` |
| `vib.wavefunctions` | PODVR 振动波函数，`(n_podvr, vmax+1)` |
| `blocks` | 固定 `(v, j, epsilon)` 的 `FSLevelBlock` 元组 |
| `energy_zero` | 从散射阈值减去的绝对能量 |
| `two_lambda_abs`, `two_S` | 电子态的 `2|Λ|` 和 `2S` |

每个 `FSLevelBlock` 的 `energies[tau]` 是绝对本征能；`coefficients[:, tau]` 是宇称基中的本征矢；`transform` 形状为 `(n_primitive, n_parity)`，把有符号 primitive basis 变换到单体宇称基。

### 5.4 精细结构通道

通常由 `build_ScattSystem` 自动构造；也可以直接调用：

```python
build_fs_channels(
    monomer: FSMonomerBasis,
    two_J: int,
    system_parity: int,
    *,
    E_cut: float = float("inf"),
    two_K_cut: int | None = None,
) -> FSChannelBasis
```

每个 `FSChannel` 用 `(block, tau, two_K, E_int)` 标记，其中 `block` 指向 `monomer.blocks`，`E_int` 已减去 `monomer.energy_zero`。通道按 `(E_int, block, tau, two_K)` 排序。

`FSChannelBasis` 的主要成员和属性为：

- `channels`：完整通道元组。
- `monomer`：对应的 `FSMonomerBasis`。
- `two_J`、`system_parity`：守恒块标签。
- `Jtot`：由 `two_J / 2` 得到的物理总角动量。
- `E_int`：形状 `(n_channel,)` 的阈值数组。
- `open_closed(Etot)`：开放/关闭通道分类。

### 5.5 精细结构完整流程

```python
from pathlib import Path

import numpy as np
import pyticc as ticc

root = Path("example/HeOH_2Pi")
pes = ticc.load_fortran_lambda_pes(root / "pes/pes.toml")
if pes.monomer_Y is None:
    raise RuntimeError("PES 未提供孤立双原子势")

mass_he, mass_o, mass_h = ticc.element_masses_au("He", "O", "H")
try:
    oh = ticc.prepare_fs_monomer(
        pes.monomer_Y,
        r=(1.4, 2.8),
        n_dvr=80,
        n_podvr=5,
        vmax=0,
        mass=ticc.reduced_mass(mass_o, mass_h),
        two_j_values=(1, 3, 5),
        two_lambda_abs=2,
        two_S=1,
        constants=root / "constant_2Pi_OH.csv",
    )
    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        oh,
        scattering_type="A+BC_fine_structure",
        two_J=1,
        system_parity=1,
        channel=ticc.ChannelSpec(E_Y_cut=100.0 * ticc.CM2AU, K_cut=None),
        potential=pes,
        reduced_mass=ticc.reduced_mass(mass_he, mass_o + mass_h),
    )
    potential_grid = ticc.prepare_potential(
        system,
        (4.0, 7.0, 12.0, 40.0),
        (0.05, 0.10, 0.25),
        n_theta=20,
    )
    result = ticc.solve(
        system,
        np.arange(10.0, 30.1, 4.0) * ticc.CM2AU,
        potential_grid,
        ticc.Propagation(),
    )
finally:
    pes.close()
```

其他已实现电子态示例包括 `example/ArO2_3Sigma-`、`example/HeNH_3Pi`、`example/ArNO_3D_2Pi` 和 `example/ArCH_2Delta`。

### 5.6 两个精细结构双原子

场自由 `AB+CD_fine_structure` 使用固定总角动量 `two_J=2J` 和总空间宇称
`system_parity=P`。两个单体角动量先耦合为 $j_{12}$，再与端对端转动耦合为 $J$。
通常由 `build_ScattSystem` 自动构造通道；对应的低层公开函数为：

```python
build_fs_diatom_diatom_channels(
    monomer_X,
    monomer_Y,
    two_J,
    system_parity,
    *,
    E_X_cut=float("inf"),
    E_Y_cut=float("inf"),
    two_K_cut=None,
) -> FSDiatomDiatomBasis
```

每个 `FSDiatomDiatomChannel` 保存 `(block_X,tau_X,block_Y,tau_Y,two_j12,two_K,E_int)`。
目前只支持 exact CC；外电场或外磁场下总 $J$ 不再守恒，不能使用这一场自由接口。

完整 Python 流程为：

```python
system = ticc.build_ScattSystem(
    monomer_X,
    monomer_Y,
    scattering_type="AB+CD_fine_structure",
    two_J=2,
    system_parity=1,
    channel=ticc.ChannelSpec(
        E_X_cut=500.0 * ticc.CM2AU,
        E_Y_cut=500.0 * ticc.CM2AU,
        K_cut=None,
    ),
    potential=spin_resolved_pes,  # 也可传标量 PESWrapper
    reduced_mass=collision_mass,
    magnetic_dipole_coefficient=C_dd,
)
potential_grid = ticc.prepare_potential(
    system,
    boundaries=(10.0, 20.0, 50.0),
    half_steps=(0.05, 0.20),
    n_theta_X=15,
    n_theta_Y=15,
    n_phi=12,
)
result = ticc.solve(
    system,
    total_energies,
    potential_grid,
    ticc.Propagation(mode="capture"),
)
```

`magnetic_dipole_coefficient` 的单位是 Hartree·bohr³，默认 `0.0`。非零时 Hamiltonian
解析加入电子自旋直接磁偶极项

$$
\hat V_{\rm dd}(R)=\frac{C_{\rm dd}}{R^3}
\left[-2\hat S_{Xz}\hat S_{Yz}
+\frac12\hat S_{X+}\hat S_{Y-}
+\frac12\hat S_{X-}\hat S_{Y+}\right].
$$

这里 $z$ 是沿分子间矢量的 BF 轴；程序内部完成两个单体分子轴自旋函数到 BF 轴的
Wigner 旋转、单体宇称展开和精细结构本征矢变换。该项保持有符号
$(\Lambda_X,\Lambda_Y)$ 不变；任一单体 $S=0$ 时自动为零。它是电子自旋的**磁**偶极项，
不要与通常已包含在长程电子 PES 中、由永久分子电偶极产生的电偶极–电偶极相互作用混淆。

## 6. 体系与通道

### 6.1 `ChannelSpec`

```python
ChannelSpec(
    vmin_X=0,
    vmin_Y=0,
    exchange_parity_X=0,
    exchange_parity_Y=0,
    E_X_cut=float("inf"),
    E_Y_cut=float("inf"),
    K_cut=None,
)
```

| 参数 | 含义 |
| --- | --- |
| `vmin_X/Y` | 单体保留的最小振动量子数 |
| `exchange_parity_X/Y` | `-1` 只保留奇数 `j`，`0` 保留全部 `j`，`1` 只保留偶数 `j` |
| `E_X_cut/E_Y_cut` | 单体内能截断，单位 Hartree |
| `K_cut` | 最大螺旋度；`None` 表示保留角动量允许的全部 `K` |

对多电子态双原子，`vmin` 和 `exchange_parity` 可以传入每个电子态一个值的元组。

验证规则：`vmin` 必须为非负整数；交换宇称只能为 `-1/0/1`；能量截断不能是 NaN 或负无穷；`K_cut` 为 `None` 或非负整数。对于精细结构通道，`E_Y_cut` 用作精细结构阈值截断，`K_cut` 会在内部转换成 `two_K_cut = 2*K_cut`；`vmin_Y` 和 `exchange_parity_Y` 不参与精细结构通道筛选。

### 6.2 `build_ScattSystem`

```python
build_ScattSystem(
    monomer_X,
    monomer_Y=None,
    *,
    scattering_type,
    Jtot=None,
    two_J=None,
    system_parity=None,
    M=None,
    channel=None,
    jmax=None,
    lmax=None,
    approx=Approx.EXACT,
    K_delta=1,
    potential=None,
    total_potential=None,
    reduced_mass=None,
    magnetic_dipole_coefficient=0.0,
) -> ScattSystem
```

主要组合方式如下：

| `scattering_type` | `monomer_X` | `monomer_Y` | 守恒量/附加参数 | PES 参数 |
| --- | --- | --- | --- | --- |
| `A+BC` | `AtomSpec` | `DiatomBasis` | `Jtot`、`system_parity` | `potential`、`reduced_mass` |
| `A+BC_diabatic` | `AtomSpec` | `DiabaticDiatomBasis` | `Jtot`、`system_parity` | `potential`、`reduced_mass` |
| `AB+CD` | `DiatomBasis` | `DiatomBasis` | `Jtot`、`system_parity` | `potential`、`reduced_mass` |
| `A+BCD` | `AtomSpec` | `TriatomBasis` | `Jtot`、`system_parity` | `potential`、`reduced_mass` |
| `A+BC_electric` | `AtomSpec` | `DiatomElectricBasis` | `M`、`lmax` | `potential`、`reduced_mass` |
| `A+BC_fine_structure` | `AtomSpec` | `FSMonomerBasis` | `two_J`、`system_parity` | `LambdaPES`、`reduced_mass` |
| `AB+CD_fine_structure` | `FSMonomerBasis` | `FSMonomerBasis` | `two_J`、`system_parity` | `PESWrapper` 或 `SpinResolvedDiatomDiatomPES`、`reduced_mass`、可选 `magnetic_dipole_coefficient` |
| `A+BC_Delves` | `DelvesMonomer` | 省略 | `Jtot`、`system_parity`、`jmax` | `total_potential` |

近似方法为 `Approx.EXACT`、`Approx.CS` 或 `Approx.NNCC`。`K_delta` 仅控制 NNCC 中相邻 `K` 的耦合窗口。非绝热、电场、精细结构和 Delves 反应散射目前只支持 exact CC。

参数作用域和互斥关系：

- 普通场自由体系传 `Jtot`；精细结构体系传 `two_J`；两者不能同时传入。
- `scattering_type` 必填，可传上表字符串或对应的 `ScatteringType` 枚举成员；程序不再根据单体类型猜测计算类型。
- `system_parity` 只能取 `-1` 或 `1`。
- 电场体系使用 `M` 和 `lmax`，不使用 `Jtot/system_parity`。
- 非反应体系使用 `potential`，Delves 体系使用 `total_potential`，不能混用。
- `reduced_mass` 必须为正；Delves 的超径质量从三个原子质量推导，禁止显式传入。
- `magnetic_dipole_coefficient` 必须为有限数，单位 Hartree·bohr³；当前仅在 `AB+CD_fine_structure` Hamiltonian 中使用。
- `channel=None` 等价于 `ChannelSpec()`。对 Delves 计算，`E_Y_cut` 必须为有限值，`exchange_parity_Y` 必须为标量。
- 返回对象的 `basis` 已经构造完成，`system.n_channel` 等于 `system.basis.n_channel`。

## 7. 势能格点与 Hamiltonian

公开的高层流程先准备完整原始势能格点：

```python
potential_grid = ticc.prepare_potential(
    system,
    boundaries,
    half_steps,
    processes=1,
    **geometry_quadrature,
)
```

`PotentialGrid` 保存 sector、全部径向端点/中点、内部坐标求积点和原始 PES 值；其 `Rmatch` 是最后一个径向边界。格点不变时，同一个对象可用于多个能量，PES 不会在传播期间重复调用。`processes` 只作用于这一步，Fortran 工作进程在批量求值结束后关闭。GPU 传播首次使用该对象时，会把完整的原始 PES 格点复制到所选 GPU 并驻留显存；后续径向窗口直接在设备端切片和收缩，不再逐窗口传输原始 PES。CPU 传播仍直接使用主存中的格点。

势能格点制备始终输出开始和完成信息。Fortran 径向批处理还会按约 10% 的间隔输出真正完成的径向点数、当前 `R` 和累计 wall time，例如：

```text
Potential preparation started: type=A+BC_diabatic, radial_points=1631, processes=4
Potential: 164/1631 radial points, R=1.128000 bohr, wall=12.418 s
Potential: 328/1631 radial points, R=1.456000 bohr, wall=24.731 s
Potential preparation complete: type=A+BC_diabatic, radial_points=1631, wall=123.902 s
```

这些进度信息属于势能制备，不受 `Propagation.print_verbose` 控制。用户提供的单次矢量化 `interaction_many` 在函数返回前无法报告内部进度，因此只输出开始和完成信息。

`prepare_potential` 根据 `system.scattering_type` 调用对应的内部格点实现。用户不需要导入 `atom_diatom`、`diabatic_atom_diatom` 等几何模块；这些模块保留为矩阵开发和测试所需的低层接口。

固定排列散射的 Hamiltonian 由 `solve(system, Etot, potential_grid, propagation)` 在内部构造。各几何模块仍保留 `build_hamiltonian(..., potential_grid=...)` 作为低层接口，主要用于矩阵检查和开发测试，不建议用户手工修改其内部回调。

各 `n_theta*` 和 `n_phi` 参数是相应角坐标的求积阶数；增大它们通常提高势能矩阵积分精度，同时按势能网格大小增加内存和计算量。`delta_symmetry=True` 使用电场方位差的对称求积。Delves 反应散射目前仍使用 `delves.build_hamiltonian` 和显式 `build_radial_sectors` 的低层流程。

## 8. 传播与求解

### 8.1 `Propagation`

```python
Propagation(
    mode: Literal["inelastic", "capture"] = "inelastic",
    memory_mb: float = 512.0,
    device: str = "auto",
    print_verbose: bool = False,
)
```

- `mode="inelastic"` 使用非弹性散射内边界；`"capture"` 使用捕获边界。
- `memory_mb` 是传播临时数组的目标内存上限，单位 MiB。
- `device` 接受 `"auto"`、`"cpu"`、`"gpu"` 或带设备编号的 `"cpu:N"`、`"gpu:N"`。`auto` 优先 GPU、无 GPU 时回退到 CPU；显式平台或编号不可用时直接报错。
- `print_verbose=True` 输出 INFO 级传播进度。

径向 `boundaries` 和 `half_steps` 属于 `PotentialGrid` 的制备参数。实际区间会被离散成固定 sector，`half_steps` 是名义半步长，并不要求区间长度恰好是其整数倍。`memory_mb` 控制传播矩阵批次的目标临时内存，不是进程总内存硬上限。`mode="capture"` 只改变内边界 LogD 初始化；传播格点、Hamiltonian 和后续结果类型不因该字符串自动改变。

### 8.2 `solve`

```python
solve(
    system: ScattSystem,
    Etot,
    potential_grid: PotentialGrid,
    propagation: Propagation,
) -> ScatteringResult | CoupledStatesResult
```

`Etot` 可以是一维 array-like，也可以是一列能量的文本文件路径；Python 接口中的能量单位为 Hartree。

输入能量必须是一维、非空、有限数组。文件由 `numpy.loadtxt` 读取，允许以 `#` 开头的注释。求解器不会排序能量，结果保持输入顺序。

返回类型由体系和近似方法决定：

| 情况 | 返回类型 |
| --- | --- |
| exact 非反应或电场散射 | `ScatteringResult` |
| CS/NNCC 非反应散射 | `CoupledStatesResult` |
| Delves 反应散射低层入口 | `ReactiveScatteringResult` |

低层兼容入口为 `solve(hamiltonian, Etot, radial_sectors, propagation)`，供 Delves 和开发测试使用。

`solve` 同时记录 wall-clock 与当前进程 CPU 时间。PES 预编译、单体基构造、势能格点计算和 Hamiltonian 预处理不计入 `result.timing`。

## 9. 结果对象

### 9.1 `ScatteringResult`

常用成员：

- `basis`：传播通道基组。
- `Etot`：总能量，形状 `(n_energy,)`。
- `Y_propagated`：传播表象的最终 LogD 矩阵。
- `Y_asymptotic`：渐近表象的 LogD 矩阵属性。
- `Smat`：每个能量一个开放通道 S 矩阵。
- `open_channel_indices`：每个能量的开放通道全局索引。
- `open_closed`：开放/关闭通道分类。
- `timing`：`wall_seconds` 和 `cpu_seconds`。

完整形状约定：

| 成员/属性 | 形状 |
| --- | --- |
| `Etot` | `(n_energy,)` |
| `Y_propagated` | `(n_energy, n_channel, n_channel)` |
| `asymptotic_transform` | `(n_channel, n_channel)` |
| `Y_asymptotic` | `(n_energy, n_channel, n_channel)`，按访问时计算 |
| `L` | `(n_channel,)`，渐近轨道角动量 |
| `open_closed.open_mask` | `(n_energy, n_channel)` |
| `open_closed.n_open/n_closed` | `(n_energy,)` |
| `open_channel_indices[i]` | `(n_open[i],)` |
| `Smat[i]` | `(n_open[i], n_open[i])` |

S 矩阵局部行列需要通过开放通道索引映射回全局基：

```python
i_energy = 0
global_indices = result.open_channel_indices[i_energy]
S = result.Smat[i_energy]

incoming_global = int(global_indices[0])
outgoing_global = int(global_indices[1])
amplitude = S[1, 0]  # outgoing <- incoming
incoming_channel = result.basis[incoming_global]
outgoing_channel = result.basis[outgoing_global]
```

`asymptotic_transform` 是传播表象到渐近表象的正交变换。不要假设 `Smat[i]` 的行列直接等于 `basis` 的前 `n_open` 个通道；必须使用 `open_channel_indices[i]`。

### 9.2 `CoupledStatesResult`

CS 和 NNCC 的结果按 K 分块保存：

- `basis`：完整 body-fixed 通道基组。
- `Etot`：总能量。
- `approx`：`Approx.CS` 或 `Approx.NNCC`。
- `blocks`：`KBlockResult` 元组。
- `timing`：求解计时。

每个 `KBlockResult` 提供 `block`、`open_channel_indices`、`Y_BF`、`Y_asymptotic`、`Smat_asymptotic` 和 `Smat_BF`。NNCC 各分块尚不能视为一个全局幺正 S 矩阵，因此接口不会自动把它们拼成单个矩阵。

`KBlockResult.block.channel_indices` 是该块在完整 `basis` 中包含的全局通道索引；`owned_channel_indices` 表示该重叠窗口负责输出的通道。`Smat_asymptotic[i]` 在该块的渐近基中，`Smat_BF[i]` 则变换回 K 标记的 BF 开放通道顺序。CS/NNCC 的 `CoupledStatesResult` 没有顶层 `Smat` 属性。

### 9.3 `ReactiveScatteringResult`

常用成员：

- `basis.qns`：渐近通道标签 `(arrangement, v, j, K)`。
- `Etot`、`Y_propagated`、`Y_asymptotic` 和 `Smat`。
- `rho_final`：物理匹配超半径。
- `surface_rho`：最后一个绝热 surface basis 的超半径。
- `radial_points`：实际传播 sector 端点。
- `energy_zero`：Hamiltonian 从原生 PES 中减去的能量；将它加回存储能量即可恢复 PES 原生能量约定。

反应散射中，最终绝热 surface 数与渐近排布通道数可以不同，因此 `Y_propagated` 的末两维为 `n_surface`，`Y_asymptotic` 的末两维为 `n_channel`，两者不能按普通固定通道结果等同处理。`basis.qns[i]`、`basis.energies[i]` 与全局渐近通道 `i` 一一对应。

## 10. 文本报告接口

`pyticc.report` 中的函数返回字符串，不直接写文件：

```python
ticc.report.rovib_levels(diatom_basis)
ticc.report.fine_structure_levels(fs_monomer)
ticc.report.channels(result.basis)
ticc.report.open_closed(result.basis, result.Etot)
ticc.report.k_blocks(result.blocks)
ticc.report.smatrix(result)
```

`report.channels` 和 `report.open_closed` 支持普通场自由通道、Electric-SF 通道、精细结构通道和 Delves 渐近排布通道。所有报告函数只返回 `str`，由调用方决定打印、写文件或写日志。

`report.smatrix` 按 `S[out,in]` 输出，初态筛选矩阵列，带 `_prime` 的末态筛选矩阵行：

| 结果类型 | 初态筛选参数 | 末态筛选参数 |
| --- | --- | --- |
| 原子–双原子 | `state`, `v`, `j` | `state_prime`, `v_prime`, `j_prime` |
| 双原子–双原子 | `state_X`, `v_X`, `j_X`, `state_Y`, `v_Y`, `j_Y`, `j_couple` | 对应的 `_prime` 参数 |
| Electric-SF | `alpha`, `m`, `l`, `m_l` | 对应的 `_prime` 参数 |
| Delves | `arrangement`, `v`, `j`, `K` | 对应的 `_prime` 参数 |
| 精细结构 | 当前输出全部开放 `(v,j,tau,epsilon,L)` 通道 | 当前仅支持 `energy_indices` |

所有结果均可用 `energy_indices` 选择能量；CS/NNCC 结果还可用 `block_index` 选择 K 分块。双原子–双原子体系不接受含义不明确的单体参数 `v` 和 `j`，必须明确使用 `_X`、`_Y` 后缀。例如：

```python
ticc.report.smatrix(
    result,
    energy_indices=0,
    v_X=0,
    j_X=0,
    v_Y=0,
    j_Y=1,
    v_X_prime=0,
    j_X_prime=2,
    v_Y_prime=0,
    j_Y_prime=1,
)
```

## 11. TOML 输入结构

`ticc.run()` 当前只覆盖四类非反应计算；精细结构、原子–三原子和 Delves 反应散射必须使用 Python 组合式接口。

### 11.1 顶层字段

公共顶层字段为：

```toml
type = "A+BC"
Jtot = 0
system_parity = 1
energies_cm = [100.0, 300.0, 500.0]
```

`energies_cm` 可以是数值数组，也可以是一列波数能量的文本文件路径。TOML 入口会自动从 cm⁻¹ 转换为 Hartree。

- `type` 必填，且大小写敏感。
- 场自由计算需要非负整数 `Jtot` 和 `system_parity = -1/1`。
- 电场计算用整数 `M` 代替 `Jtot/system_parity`。
- `atom` 是单个元素符号字符串；`diatom`、`diatom_X`、`diatom_Y` 必须恰有两个元素符号。

### 11.2 公共配置段

公共配置段：

```toml
[approximation]
method = "exact" # exact、cs 或 nncc
K_delta = 1      # 仅 NNCC 使用

[pes]
path = "pes"
sources = ["interaction-PES.f"]
wrapper = "pyticc_wrapper.f90"
workdir = "."
lapack = false

[potential_grid]
radial_boundaries = [4.5, 6.5, 8.0, 12.0]
radial_half_steps = [0.05, 0.08, 0.10]
processes = 4

[propagation]
mode = "inelastic"
memory_limit_mb = 512.0
device = "auto"
print_verbose = false
```

默认值和路径规则：

| 字段 | 是否必填 | 默认值/说明 |
| --- | --- | --- |
| `[approximation].method` | 整段可省略 | 省略整段时为 `exact`；取值 `exact/cs/nncc`，不区分大小写 |
| `[approximation].K_delta` | 否 | `1`；NNCC 的相邻 K 范围 |
| `[pes].path` | 否 | `.`，相对于主输入 TOML |
| `[pes].sources` | 否 | `["interaction-PES.f"]`，相对于 `[pes].path` |
| `[pes].wrapper` | 否 | `pyticc_wrapper.f90`，相对于 `[pes].path` |
| `[pes].workdir` | 否 | `.`，相对于 `[pes].path` |
| `[pes].lapack` | 否 | `false` |
| `[pes].n_state` | 仅非绝热、可选 | `2` |
| `[potential_grid].radial_boundaries` | 是 | `N+1` 个 bohr 值 |
| `[potential_grid].radial_half_steps` | 是 | `N` 个 bohr 值 |
| `[potential_grid].processes` | 否 | `1`；仅用于这次 PES 格点计算 |
| `[propagation].mode` | 是 | `inelastic` 或 `capture` |
| `[propagation].memory_limit_mb` | 否 | `512.0` |
| `[propagation].device` | 否 | `auto` |
| `[propagation].print_verbose` | 否 | `false` |

### 11.3 各计算类型字段

不同 `type` 所需的体系字段和专用段为：

| `type` | 体系字段 | 基组段 | 求积段 | 通道能量截断 |
| --- | --- | --- | --- | --- |
| `A+BC` | `atom`、`diatom`、`Jtot`、`system_parity` | `[basis]` | `n_theta` | `E_Y_cut_cm` |
| `A+BC_diabatic` | 同上 | `[basis]`，截断可按电子态给数组 | `n_theta` | `E_Y_cut_cm` |
| `AB+CD` | `diatom_X`、`diatom_Y`、`Jtot`、`system_parity` | `[basis_X]`、`[basis_Y]` | `n_theta_X`、`n_theta_Y`、`n_phi` | `E_X_cut_cm`、`E_Y_cut_cm` |
| `A+BC_electric` | `atom`、`diatom`、`M` | `[basis]` 和 `[electric]` | `n_theta_r`、`n_theta_R`、`n_delta` | `E_Y_cut_cm` |

普通双原子 `[basis]` 段包含 `r`、`n_dvr`、`n_podvr`、`vmax` 和 `jmax`。场修饰双原子还包含 `lmax`、`n_alpha`，并用 `[electric]` 的 `strength_au` 和 `response_csv` 指定电场与响应数据。

非电场 `[channels]` 段需要 `K_cut`，其值为非负整数或字符串 `"none"`；`vmin_X/Y` 和 `exchange_parity_X/Y` 的默认值为 `0`。

绝热原子–双原子的专用部分：

```toml
[basis]
r = [1.5, 4.5]
n_dvr = 100
n_podvr = 5
vmax = 3
jmax = 20

[quadrature]
n_theta = 35

[channels]
vmin_Y = 0
exchange_parity_Y = 0
E_Y_cut_cm = 2000.0
K_cut = "none"
```

非绝热原子–双原子的 `n_podvr/vmax/jmax/vmin_Y/exchange_parity_Y` 可为一个共享整数，也可为长度等于 `[pes].n_state` 的数组，例如 `jmax = [55, 56]`。其 `[approximation].method` 必须为 `exact`。

双原子–双原子使用相同结构的 `[basis_X]` 和 `[basis_Y]`，并在 `[channels]` 中分别提供 `E_X_cut_cm`、`E_Y_cut_cm`。求积段必须提供 `n_theta_X`、`n_theta_Y` 和 `n_phi`。

电场原子–双原子的专用部分：

```toml
[basis]
r = [1.5, 4.5]
n_dvr = 100
n_podvr = 5
jmax = 8
lmax = 1
n_alpha = 3

[electric]
strength_au = 1.0e-3
response_csv = "pes/HF_ele.csv"

[quadrature]
n_theta_r = 16
n_theta_R = 16
n_delta = 16
delta_symmetry = true

[channels]
E_Y_cut_cm = 2000.0
```

Electric-SF 不读取 `K_cut`，且只允许 exact CC。`response_csv` 相对于主输入 TOML 解析。

### 11.4 捕获边界

捕获计算不使用新的 `type`，而是在普通几何输入中设置：

```toml
[potential_grid]
radial_boundaries = [20.0, 25.0, 60.0, 100.0]
radial_half_steps = [0.05, 0.5, 1.0]

[propagation]
mode = "capture"
```

完整的 NNCC 双原子–双原子捕获示例见 `example/K2Rb2_capture/input.toml`。

### 11.5 完整示例

可运行的完整 TOML 示例位于：

- `example/ArHF/input.toml`
- `example/H2HF/input.toml`
- `example/HO2_diabatic/input.toml`
- `example/ArHF_electric/input.toml`
- `example/K2Rb2_capture/input.toml`

## 12. 单位、质量与辅助函数

### 12.1 能量与频率

```python
energy_to_au(value: float, unit: EnergyUnit) -> float
```

`EnergyUnit` 是字面量类型 `"au" | "cm-1" | "Hz" | "kHz" | "MHz" | "GHz"`。不支持的单位名会抛出 `KeyError`；该函数面向标量，数组应直接乘以对应换算常数。

### 12.2 原子质量

```python
element_mass_au(symbol: str) -> float
element_masses_au(*symbols: str) -> tuple[float, ...]
reduced_mass(mass1: float, mass2: float) -> float
```

内置元素符号目前包括 `H`、`D`、`He`、`Li`、`C`、`N`、`O`、`F`、`S`、`Cl`、`Ar`、`K` 和 `Rb`。质量以电子质量返回。`H`、`D`、`C` 等采用具体同位素质量；需要其他同位素或更精确的体系质量时，应由调用方直接提供。

`reduced_mass(m1, m2)` 计算 $\mu=m_1m_2/(m_1+m_2)$。两个输入应使用相同单位；PyTICC 的散射和单体基接口要求结果为原子单位质量。

## 13. 错误处理和资源管理

- 配置错误、维度不匹配、非有限数值和不支持的物理组合通常抛出 `ValueError`。
- PES 类型与计算类型不匹配通常抛出 `TypeError`。
- PES 源文件或运行目录不存在时抛出 `FileNotFoundError`。
- 编译失败时抛出 `RuntimeError`，错误信息中给出构建日志路径。
- 用户自行加载的 Fortran PES 应在 `finally` 中调用 `close()`。

`PESWrapper`、`DiabaticPESWrapper`、`LambdaPES` 和 `TotalPES` 的 `close()` 用于释放加载器创建的持久资源。纯 Python 回调构造的对象通常没有资源需要释放，但仍可安全调用。不要在求解进行中关闭 PES。

PyTICC 使用 `loguru` 输出诊断信息。异常消息通常包含出错字段、实际形状或源文件路径；调用方不应依赖完整英文消息做控制流判断，应按异常类型处理。

## 14. 公开 API 速查

本节按 `pyticc.__all__` 汇总顶层公开名称。表中“构造产物”表示通常由 builder 或 solver 返回，用户一般不直接实例化。

### 14.1 运行、体系与求解

| 名称 | 签名/用途 |
| --- | --- |
| `run` | `run(source, *, pes=None)`：运行四类 TOML 非反应输入 |
| `build_ScattSystem` | `build_ScattSystem(monomer_X, monomer_Y=None, *, scattering_type, Jtot=None, two_J=None, system_parity=None, M=None, channel=None, jmax=None, lmax=None, approx=Approx.EXACT, K_delta=1, potential=None, total_potential=None, reduced_mass=None, magnetic_dipole_coefficient=0.0)` |
| `build_fs_hamiltonian` | `build_fs_hamiltonian(system, *, n_theta=24)` |
| `prepare_potential` | `prepare_potential(system, boundaries, half_steps, *, processes=1, **quadrature)` |
| `solve` | `solve(system, Etot, potential_grid, propagation)` |
| `Approx` | 枚举：`EXACT`、`CS`、`NNCC`；字符串值分别为 `exact/cs/nncc` |
| `ScatteringType` | 枚举：`A+BC`、`A+BC_electric`、`A+BC_fine_structure`、`A+BC_diabatic`、`A+BC_Delves`、`AB+CD`、`AB+CD_fine_structure`、`A+BCD` |
| `ChannelSpec` | 通道筛选 dataclass，详见第 6.1 节 |
| `Propagation` | 传播模式、内存和设备配置 dataclass，详见第 8.1 节 |
| `PotentialGrid` | 径向/求积格点和原始 PES 值的不可变容器 |
| `ScattSystem` | 已准备体系的不可变 dataclass；通常由 `build_ScattSystem` 返回 |
| `ScattHamiltonian` | 固定通道 Hamiltonian；通常由几何 builder 返回 |
| `DelvesHamiltonian` | Delves 绝热 surface Hamiltonian；通常由 `delves.build_hamiltonian` 返回 |

### 14.2 PES

| 名称 | 签名/用途 |
| --- | --- |
| `PESWrapper` | 标量单体/相互作用 PES 回调容器 |
| `DiabaticPESWrapper` | `DiabaticPESWrapper(n_state, monomer, interaction, interaction_many=None)` |
| `LambdaPES` | `LambdaPES(interaction, monomer_Y=None, interaction_many=None)` |
| `OrbitalState` | 一个有符号 $(2\Lambda_X,2\Lambda_Y)$ 轨道乘积态 |
| `SpinResolvedDiatomDiatomPES` | `SpinResolvedDiatomDiatomPES(interaction, two_total_spins, orbital_states, interaction_many=None)` |
| `TotalPES` | `TotalPES(potential)` |
| `as_lambda_pes` | `as_lambda_pes(pes: PESWrapper) -> LambdaPES` |
| `as_spin_resolved_diatom_diatom_pes` | 把标量 2+2 PES 提升为所有总自旋面相同、orbital identity 的表示 |
| `load_fortran_pes` | 标量 Fortran PES 加载器 |
| `load_fortran_diabatic_pes` | 多电子态非绝热 Fortran PES 加载器 |
| `load_fortran_lambda_pes` | Λ 分辨 Fortran PES 加载器 |
| `load_fortran_total_pes` | 三原子总 PES 加载器 |

四个 Fortran 加载器都接受 `sources`、可选 `wrapper`、`workdir` 和 `lapack`；非绝热加载器额外接受 `n_state`。并行进程数在势能格点制备时传入。

### 14.3 单体和基组

| 名称 | 签名/用途 |
| --- | --- |
| `AtomSpec` | `AtomSpec()`：无内部结构原子 |
| `build_SineDVR` | `build_SineDVR(a, b, n_dvr, mass, pot_func)` |
| `prepare_Diatom` | 从标量单体势一步构造 `DiatomBasis` |
| `build_DiatomBasis` | 从已有 `SineDVR` 构造 `DiatomBasis` |
| `prepare_DiabaticDiatom` | 从所有电子态单体势一步构造 `DiabaticDiatomBasis` |
| `build_DiabaticDiatomBasis` | 从每个电子态的 `SineDVR` 序列构造多态基 |
| `prepare_DiatomElectric` | 从单体势和响应表一步构造 `DiatomElectricBasis` |
| `build_DiatomElectricBasis` | 从已有 `SineDVR` 和响应表构造电场基 |
| `prepare_Triatom` | 从一个物理三原子 PES 自动构造参考 DVR 和收缩转振基 |
| `build_TriatomBasis` | 从已有两个 `SineDVR` 构造收缩三原子转振基 |
| `prepare_Delves` | 准备 Delves 三排布的质量和能量零点信息 |
| `prepare_fs_monomer` | 从单体势、光谱常数和量子数构造 `FSMonomerBasis` |
| `build_fs_channels` | 从 `FSMonomerBasis` 显式构造固定 `(J,P)` 通道 |
| `build_fs_diatom_diatom_channels` | 从两个 `FSMonomerBasis` 构造场自由固定 `(J,P)` 的 2+2 精细结构通道 |

作为构造产物公开的类型包括 `RovibBasis`、`DiatomBasis`、`DiabaticDiatomBasis`、`DiatomElectricBasis`、`TriatomBasis`、`DelvesMonomer`、`DelvesBasis`、`FSMonomerBasis` 和 `FSDiatomDiatomBasis`。除编写测试或自定义低层 builder 外，优先使用对应的 `prepare_*`/`build_*` 函数，以确保形状、能量零点和量子数可用性一致。

### 14.4 精细结构与电场数据

| 名称 | 用途 |
| --- | --- |
| `FSConstants` | 一组 Hartree 制有效分子常数；支持 `FSConstants.from_unit(...)` |
| `FSConstantsTable` | 按 `v` 保存的不可变常数表；使用 `vibrational_levels` 和 `for_v(v)` |
| `load_fs_constants_csv` | 加载 `v,constant,value,unit` 长格式 CSV |
| `FSDiatomDiatomChannel` | 一个 2+2 精细结构通道的单体能级、$j_{12}$、$K$ 和阈值标签 |
| `FSDiatomDiatomBasis` | 固定场自由 $(J,P)$ 的两个精细结构双原子通道基 |
| `ElectricResponseTable` | 双原子电响应径向表；`evaluate(r)` 做无外推自然样条插值 |
| `load_electric_response_csv` | 加载固定六列的原子单位电响应 CSV |

### 14.5 结果类型

| 名称 | 产生条件 |
| --- | --- |
| `ScatteringResult` | exact 非反应、Electric-SF 或精细结构散射 |
| `CoupledStatesResult` | CS/NNCC 非反应散射 |
| `ReactiveScatteringResult` | Delves 三原子反应散射 |

所有结果均保留输入能量顺序。exact 和反应结果的 `Smat` 均按每个能量的开放通道单独存放；CS/NNCC 结果的矩阵保存在 `blocks` 内。

### 14.6 单位、质量和报告

| 名称 | 用途 |
| --- | --- |
| `EnergyUnit` | 支持的能量/频率单位字面量类型 |
| `energy_to_au` | 单个标量转 Hartree |
| `CM2AU/AU2CM` | cm⁻¹ 与 Hartree 互换 |
| `HZ2AU/AU2HZ` | Hz 与 Hartree 互换 |
| `KHZ2AU/AU2KHZ` | kHz 与 Hartree 互换 |
| `MHZ2AU/AU2MHZ` | MHz 与 Hartree 互换 |
| `GHZ2AU/AU2GHZ` | GHz 与 Hartree 互换 |
| `ANG2AU/AU2ANG` | Å 与 bohr 互换 |
| `element_mass_au` | 一个内置元素质量 |
| `element_masses_au` | 按输入顺序返回多个质量 |
| `reduced_mass` | 两体约化质量 |
| `report` | `rovib_levels`、`fine_structure_levels`、`channels`、`open_closed`、`k_blocks`、`smatrix` 文本报告命名空间 |
