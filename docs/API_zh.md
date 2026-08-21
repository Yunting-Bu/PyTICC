# PyTICC 接口说明

本文档说明 PyTICC 0.1.x 当前可供用户调用的接口。内容以 `pyticc.__all__`、高层散射流程和现有示例为准；未在本文列出的 `pyticc.*` 子模块函数主要是内部数值实现，不保证接口稳定。

## 1. 基本约定

PyTICC 默认使用原子单位：

| 物理量 | 输入/输出单位 |
| --- | --- |
| 长度、径向传播坐标 | bohr |
| 能量、势能、通道阈值 | Hartree |
| 质量 | 电子质量 |
| 角度 | radian |
| 电场强度 | 原子单位 |

常用换算常数可直接从 `pyticc` 导入：

```python
import pyticc as ticc

energy_au = 100.0 * ticc.CM2AU
energy_cm = energy_au * ticc.AU2CM
length_au = 1.0 * ticc.ANG2AU
length_angstrom = length_au * ticc.AU2ANG
```

数组索引和形状遵循以下约定：

- 电子态编号从 `0` 开始。
- 振动、转动和螺旋度量子数分别记为 `v`、`j` 和 `K`。
- 总能量数组形状为 `(n_energy,)`。
- LogD 矩阵批次形状为 `(n_energy, n_channel, n_channel)`。
- 每个能量的开放通道数可能不同，因此 S 矩阵使用 `tuple` 保存；`Smat[i]` 的形状为 `(n_open[i], n_open[i])`。
- 势能回调函数必须返回实数、有限值，并严格满足本文给出的数组形状。

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
- 当前 TOML 入口支持 `atom-diatom`、`electric-atom-diatom`、`diabatic-atom-diatom` 和 `diatom-diatom`。
- 原子–三原子和 Delves 反应散射目前使用 Python 接口。

### 2.2 Python 组合式入口

推荐的 Python 调用流程为：

1. 构造或加载 PES。
2. 准备单体基组。
3. 用 `build_ScattSystem` 构造物理体系和通道。
4. 从相应几何模块构造 Hamiltonian。
5. 用 `solve` 传播并匹配。

以绝热原子–双原子散射为例：

```python
from pathlib import Path

import numpy as np

import pyticc as ticc
from pyticc.scattering import atom_diatom

pes_dir = Path("pes")
pes = ticc.load_fortran_pes(
    [pes_dir / "interaction-PES.f"],
    pes_dir / "pyticc_wrapper.f90",
    workdir=pes_dir,
    processes=4,
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
        Jtot=0,
        system_parity=1,
        channel=ticc.ChannelSpec(E_Y_cut=2000.0 * ticc.CM2AU),
        potential=pes,
        reduced_mass=ticc.reduced_mass(mass_ar, mass_hf),
    )
    hamiltonian = atom_diatom.build_hamiltonian(system, n_theta=35)
    result = ticc.solve(
        hamiltonian,
        np.array([100.0, 300.0, 500.0]) * ticc.CM2AU,
        ticc.Propagation(
            boundaries=(4.5, 6.5, 8.0, 12.0),
            half_steps=(0.05, 0.08, 0.10),
        ),
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

### 3.3 `TotalPES`

```python
TotalPES(potential)
```

Delves 三原子反应散射使用总势能面。`potential(bonds)` 接收形状 `(3, n_grid)`、行顺序为 `(r_AB, r_BC, r_CA)` 的三个物理键长，返回 `(n_grid,)` 的总能量。

该接口要求势能面直接返回一个绝热电子面上的总能量，不得减去双原子势、额外加入单体势或自行改变能量零点。

### 3.4 Fortran PES 加载

```python
load_fortran_pes(sources, wrapper=None, *, workdir=None, processes=1, lapack=False)
load_fortran_diabatic_pes(
    sources, wrapper=None, *, n_state=2, workdir=None, processes=1, lapack=False
)
load_fortran_total_pes(sources, wrapper=None, *, workdir=None, lapack=False)
```

- `sources` 可为一个或多个 Fortran 源文件，也可为描述源文件和 wrapper 的短 TOML 文件。
- `workdir` 是 PES 运行时数据文件所在目录。
- `processes` 控制批量径向求值使用的进程数。
- `lapack=True` 表示 PES 编译时需要链接 LAPACK。
- 加载结果持有编译模块或工作进程时，应在使用结束后调用 `close()`；重复调用是安全的。

Fortran wrapper 的必需子程序为：

| 类型 | 必需子程序 | 可选子程序 |
| --- | --- | --- |
| 绝热相互作用 PES | `pyticc_interaction_grid` | `pyticc_monomer_x_grid`、`pyticc_monomer_y_grid` |
| 非绝热 PES | `pyticc_diabatic_interaction_grid`、`pyticc_diabatic_monomer_grid` | 无 |
| 三原子总 PES | `pyticc_total_grid` | 无 |

具体参数顺序可参考 `example/ArHF/pes/pyticc_wrapper.f90`、`example/HO2_diabatic/pes/pyticc_wrapper.f90` 和 `example/H3_Delves/pyticc_total_wrapper.f90`。

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

需要复用原始 DVR 时，可先调用 `build_SineDVR`，再调用 `build_DiatomBasis`。

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

### 4.5 原子–三原子和 Delves 基组

原子–三原子单体基组使用 `build_TriatomBasis(...) -> TriatomBasis`。该接口接收两个径向 DVR、三原子内坐标势能、质量、平衡构型、角度求积和量子数截断；其参数较多，建议从测试或具体体系示例开始配置。

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

## 5. 体系与通道

### 5.1 `ChannelSpec`

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

### 5.2 `build_ScattSystem`

```python
build_ScattSystem(
    monomer_X,
    monomer_Y=None,
    *,
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
) -> ScattSystem
```

主要组合方式如下：

| 计算 | `monomer_X` | `monomer_Y` | 守恒量/附加参数 | PES 参数 |
| --- | --- | --- | --- | --- |
| 原子–双原子 | `AtomSpec` | `DiatomBasis` | `Jtot`、`system_parity` | `potential`、`reduced_mass` |
| 非绝热原子–双原子 | `AtomSpec` | `DiabaticDiatomBasis` | `Jtot`、`system_parity` | `potential`、`reduced_mass` |
| 双原子–双原子 | `DiatomBasis` | `DiatomBasis` | `Jtot`、`system_parity` | `potential`、`reduced_mass` |
| 原子–三原子 | `AtomSpec` | `TriatomBasis` | `Jtot`、`system_parity` | `potential`、`reduced_mass` |
| 电场原子–双原子 | `AtomSpec` | `DiatomElectricBasis` | `M`、`lmax` | `potential`、`reduced_mass` |
| 精细结构原子–双原子 | `AtomSpec` | `FSMonomerBasis` | `two_J`、`system_parity` | `LambdaPES`、`reduced_mass` |
| Delves 反应散射 | `DelvesMonomer` | 省略 | `Jtot`、`system_parity`、`jmax` | `total_potential` |

近似方法为 `Approx.EXACT`、`Approx.CS` 或 `Approx.NNCC`。`K_delta` 仅控制 NNCC 中相邻 `K` 的耦合窗口。非绝热、电场、精细结构和 Delves 反应散射目前只支持 exact CC。

## 6. Hamiltonian 构造

Hamiltonian 构造函数位于对应的几何子模块中：

```python
from pyticc.scattering import (
    atom_diatom,
    atom_triatom,
    delves,
    diabatic_atom_diatom,
    diatom_diatom,
    fine_structure_atom_diatom,
)
```

| 函数 | 说明 |
| --- | --- |
| `atom_diatom.build_hamiltonian(system, *, n_theta=16)` | 绝热原子–双原子 |
| `atom_diatom.build_hamiltonian_electric_sf(system, *, n_theta_r=16, n_theta_R=16, n_delta=16, delta_symmetry=True)` | 电场空间固定表象 |
| `diabatic_atom_diatom.build_hamiltonian(system, *, n_theta=16)` | 非绝热原子–双原子 |
| `diatom_diatom.build_hamiltonian(system, *, n_theta_X=15, n_theta_Y=15, n_phi=12)` | 双原子–双原子 |
| `atom_triatom.build_hamiltonian(system, *, n_theta_1=None, n_theta_2=16, n_phi=16)` | 原子–三原子 |
| `fine_structure_atom_diatom.build_hamiltonian(system, *, n_theta=24)` | 含精细结构的原子–双原子 |
| `delves.build_hamiltonian(system, *, overlap_cut=1e-4)` | Delves 三原子反应散射 |

返回值为 `ScattHamiltonian` 或 `DelvesHamiltonian`，通常直接传给 `solve`，不建议用户手工修改其内部矩阵回调。

## 7. 传播与求解

### 7.1 `Propagation`

```python
Propagation(
    boundaries: tuple[float, ...],
    half_steps: tuple[float, ...],
    mode: Literal["inelastic", "capture"] = "inelastic",
    memory_mb: float = 512.0,
    device: str = "auto",
    print_verbose: bool = False,
)
```

- `boundaries` 有 `N+1` 个严格递增的正数，定义 `N` 个径向区间。
- `half_steps` 有 `N` 个正数，分别定义每个区间的 LDMD 名义半步长。
- `mode="inelastic"` 使用非弹性散射内边界；`"capture"` 使用捕获边界。
- `memory_mb` 是传播临时数组的目标内存上限，单位 MiB。
- `device` 接受 `"auto"`、`"cpu"` 或 `"gpu"`。`auto` 无 GPU 时回退到 CPU；`gpu` 找不到 GPU 时直接报错。
- `print_verbose=True` 输出 INFO 级传播进度。

`boundaries[-1]` 同时是非反应散射的匹配距离 `Rmatch`。

### 7.2 `solve`

```python
solve(
    hamiltonian: ScattHamiltonian | DelvesHamiltonian,
    Etot,
    propagation: Propagation,
) -> ScatteringResult | CoupledStatesResult | ReactiveScatteringResult
```

`Etot` 可以是一维 array-like，也可以是一列能量的文本文件路径；Python 接口中的能量单位为 Hartree。

返回类型由 Hamiltonian 和近似方法决定：

| 情况 | 返回类型 |
| --- | --- |
| exact 非反应或电场散射 | `ScatteringResult` |
| CS/NNCC 非反应散射 | `CoupledStatesResult` |
| Delves 反应散射 | `ReactiveScatteringResult` |

## 8. 结果对象

### 8.1 `ScatteringResult`

常用成员：

- `basis`：传播通道基组。
- `Etot`：总能量，形状 `(n_energy,)`。
- `Y_propagated`：传播表象的最终 LogD 矩阵。
- `Y_asymptotic`：渐近表象的 LogD 矩阵属性。
- `Smat`：每个能量一个开放通道 S 矩阵。
- `open_channel_indices`：每个能量的开放通道全局索引。
- `open_closed`：开放/关闭通道分类。
- `timing`：`wall_seconds` 和 `cpu_seconds`。

### 8.2 `CoupledStatesResult`

CS 和 NNCC 的结果按 K 分块保存：

- `basis`：完整 body-fixed 通道基组。
- `Etot`：总能量。
- `approx`：`Approx.CS` 或 `Approx.NNCC`。
- `blocks`：`KBlockResult` 元组。
- `timing`：求解计时。

每个 `KBlockResult` 提供 `block`、`open_channel_indices`、`Y_BF`、`Y_asymptotic`、`Smat_asymptotic` 和 `Smat_BF`。NNCC 各分块尚不能视为一个全局幺正 S 矩阵，因此接口不会自动把它们拼成单个矩阵。

### 8.3 `ReactiveScatteringResult`

常用成员：

- `basis.qns`：渐近通道标签 `(arrangement, v, j, K)`。
- `Etot`、`Y_propagated`、`Y_asymptotic` 和 `Smat`。
- `rho_final`：物理匹配超半径。
- `surface_rho`：最后一个绝热 surface basis 的超半径。
- `radial_points`：实际传播 sector 端点。
- `energy_zero`：Hamiltonian 从原生 PES 中减去的能量；将它加回存储能量即可恢复 PES 原生能量约定。

## 9. 文本报告接口

`pyticc.report` 中的函数返回字符串，不直接写文件：

```python
ticc.report.rovib_levels(diatom_basis)
ticc.report.channels(result.basis)
ticc.report.open_closed(result.basis, result.Etot)
ticc.report.k_blocks(result.blocks)
ticc.report.smatrix(result)
```

`report.channels` 和 `report.open_closed` 支持普通场自由通道、Electric-SF 通道和 Delves 渐近排布通道。

`report.smatrix` 按 `S[out,in]` 输出，初态筛选矩阵列，带 `_prime` 的末态筛选矩阵行：

| 结果类型 | 初态筛选参数 | 末态筛选参数 |
| --- | --- | --- |
| 原子–双原子 | `state`, `v`, `j` | `state_prime`, `v_prime`, `j_prime` |
| 双原子–双原子 | `state_X`, `v_X`, `j_X`, `state_Y`, `v_Y`, `j_Y`, `j_couple` | 对应的 `_prime` 参数 |
| Electric-SF | `alpha`, `m`, `l`, `m_l` | 对应的 `_prime` 参数 |
| Delves | `arrangement`, `v`, `j`, `K` | 对应的 `_prime` 参数 |

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

## 10. TOML 输入结构

公共顶层字段为：

```toml
type = "atom-diatom"
Jtot = 0
system_parity = 1
energies_cm = [100.0, 300.0, 500.0]
```

`energies_cm` 可以是数值数组，也可以是一列波数能量的文本文件路径。TOML 入口会自动从 cm⁻¹ 转换为 Hartree。

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
processes = 4
lapack = false

[propagation]
radial_boundaries = [4.5, 6.5, 8.0, 12.0]
radial_half_steps = [0.05, 0.08, 0.10]
mode = "inelastic"
memory_limit_mb = 512.0
device = "auto"
print_verbose = false
```

不同 `type` 所需的体系字段和专用段为：

| `type` | 体系字段 | 基组段 | 求积段 | 通道能量截断 |
| --- | --- | --- | --- | --- |
| `atom-diatom` | `atom`、`diatom`、`Jtot`、`system_parity` | `[basis]` | `n_theta` | `E_Y_cut_cm` |
| `diabatic-atom-diatom` | 同上 | `[basis]`，截断可按电子态给数组 | `n_theta` | `E_Y_cut_cm` |
| `diatom-diatom` | `diatom_X`、`diatom_Y`、`Jtot`、`system_parity` | `[basis_X]`、`[basis_Y]` | `n_theta_X`、`n_theta_Y`、`n_phi` | `E_X_cut_cm`、`E_Y_cut_cm` |
| `electric-atom-diatom` | `atom`、`diatom`、`M` | `[basis]` 和 `[electric]` | `n_theta_r`、`n_theta_R`、`n_delta` | `E_Y_cut_cm` |

普通双原子 `[basis]` 段包含 `r`、`n_dvr`、`n_podvr`、`vmax` 和 `jmax`。场修饰双原子还包含 `lmax`、`n_alpha`，并用 `[electric]` 的 `strength_au` 和 `response_csv` 指定电场与响应数据。

非电场 `[channels]` 段需要 `K_cut`，其值为非负整数或字符串 `"none"`；`vmin_X/Y` 和 `exchange_parity_X/Y` 的默认值为 `0`。

可运行的完整 TOML 示例位于：

- `example/ArHF/input.toml`
- `example/H2HF/input.toml`
- `example/HO2_diabatic/input.toml`
- `example/ArHF_electric/input.toml`

## 11. 质量辅助函数

```python
element_mass_au(symbol: str) -> float
element_masses_au(*symbols: str) -> tuple[float, ...]
reduced_mass(mass1: float, mass2: float) -> float
```

内置元素符号目前包括 `H`、`D`、`He`、`Li`、`N`、`O`、`F`、`S`、`Cl` 和 `Ar`。其他同位素或元素应由调用方直接提供以电子质量为单位的质量。

## 12. 错误处理和资源管理

- 配置错误、维度不匹配、非有限数值和不支持的物理组合通常抛出 `ValueError`。
- PES 类型与计算类型不匹配通常抛出 `TypeError`。
- PES 源文件或运行目录不存在时抛出 `FileNotFoundError`。
- 编译失败时抛出 `RuntimeError`，错误信息中给出构建日志路径。
- 用户自行加载的 Fortran PES 应在 `finally` 中调用 `close()`，尤其是在 `processes > 1` 时。
