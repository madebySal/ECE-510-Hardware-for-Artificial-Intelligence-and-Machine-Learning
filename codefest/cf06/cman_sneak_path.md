## Circuit Setup
 
A 2×2 resistive crossbar with the following fixed cell resistances:
 
| Cell    | Resistance | State |
|---------|-----------|-------|
| R[0][0] | 1 kΩ      | ON    |
| R[0][1] | 2 kΩ      | OFF   |
| R[1][0] | 2 kΩ      | OFF   |
| R[1][1] | 1 kΩ      | ON    |
 
- **Rows** carry input voltages.  
- **Columns** carry output currents, sensed at virtual ground (0 V).  
- Low-resistance (1 kΩ) = "ON" weight; high-resistance (2 kΩ) = "OFF" weight.
---
 
## (a) Ideal I_col0
 
**Conditions:**
- V_row0 = 1 V (driven)
- V_row1 = 0 V (grounded)
- V_col0 = 0 V (virtual ground / sense node)
- V_col1 = 0 V (grounded)
With all rows and columns at fixed voltages, each cell independently obeys Ohm's Law.
 
**Current through R[0][0] into col0:**
 
```
I_R[0][0] = (V_row0 - V_col0) / R[0][0]
           = (1 V - 0 V) / 1000 Ω
           = 1.000 mA
```
 
**Current through R[1][0] into col0:**
 
```
I_R[1][0] = (V_row1 - V_col0) / R[1][0]
           = (0 V - 0 V) / 2000 Ω
           = 0 mA
```
 
No voltage difference across R[1][0] → zero contribution.
 
```
I_col0_ideal = 1.000 mA
```
 
This is the correct MVM result: col0 encodes the product of V_row0 = 1 V and conductance G[0][0] = 1/1kΩ = 1 mS.
 
---
 
## (b) KCL Solution for Floating Node Voltages V_row1 and V_col1
 
**Conditions:**
- V_row0 = 1 V (driven)
- V_col0 = 0 V (held at sense ground)
- V_row1 = ? (floating / undriven)
- V_col1 = ? (floating / undriven)
Because row1 and col1 are undriven, their voltages are set by the resistive network. We apply **Kirchhoff's Current Law (KCL)**: the algebraic sum of currents leaving each floating node equals zero.
 
---
 
### KCL at Node V_row1
 
Connections from V_row1:
- To col0 (0 V) through R[1][0] = 2 kΩ
- To col1 (V_col1) through R[1][1] = 1 kΩ
```
(V_row1 - V_col0) / R[1][0]  +  (V_row1 - V_col1) / R[1][1]  =  0
 
(V_row1 - 0) / 2000  +  (V_row1 - V_col1) / 1000  =  0
 
Multiply through by 2000:
 
V_row1  +  2*(V_row1 - V_col1)  =  0
 
3*V_row1  -  2*V_col1  =  0          ... (1)
```
 
---
 
### KCL at Node V_col1
 
Connections from V_col1:
- To row0 (1 V) through R[0][1] = 2 kΩ
- To row1 (V_row1) through R[1][1] = 1 kΩ
```
(V_col1 - V_row0) / R[0][1]  +  (V_col1 - V_row1) / R[1][1]  =  0
 
(V_col1 - 1) / 2000  +  (V_col1 - V_row1) / 1000  =  0
 
Multiply through by 2000:
 
(V_col1 - 1)  +  2*(V_col1 - V_row1)  =  0
 
3*V_col1  -  2*V_row1  =  1          ... (2)
```
 
---
 
### Solving the 2×2 Linear System
 
From equation (1):
 
```
V_col1 = (3/2) * V_row1
```
 
Substitute into equation (2):
 
```
3 * (3/2) * V_row1  -  2 * V_row1  =  1
 
(9/2) * V_row1  -  (4/2) * V_row1  =  1
 
(5/2) * V_row1  =  1
 
V_row1  =  2/5  =  0.4 V
```
 
Then:
 
```
V_col1  =  (3/2) * 0.4  =  0.6 V
```
 
### Result
 
| Floating Node | Solved Voltage |
|---------------|---------------|
| V_row1        | **0.4 V**     |
| V_col1        | **0.6 V**     |
 
The floating row1 is pulled up from 0 V to 0.4 V through resistive coupling via R[1][1]–R[0][1] back to V_row0. The floating col1 settles at 0.6 V, partway between 0 V and V_row0 = 1 V.
 
---
 
## (c) Actual I_col0 Including the Sneak Path
 
With V_row1 = 0.4 V (floating) and V_col0 = 0 V (held), two distinct current paths feed into col0:
 
**Path 1 — Intended path:** row0 (1 V) → R[0][0] (1 kΩ) → col0 (0 V)
 
```
I_intended = (V_row0 - V_col0) / R[0][0]
           = (1.0 - 0) / 1000
           = 1.000 mA
```
 
**Path 2 — Sneak path:** row1 (0.4 V, floating) → R[1][0] (2 kΩ) → col0 (0 V)
 
```
I_sneak = (V_row1 - V_col0) / R[1][0]
        = (0.4 - 0) / 2000
        = 0.200 mA
```
 
**Total actual output current:**
 
```
I_col0_actual = I_intended + I_sneak
              = 1.000 + 0.200
              = 1.200 mA
```
 
### Summary Table
 
| Current Component              | Value        | Description                         |
|-------------------------------|--------------|--------------------------------------|
| I_intended (row0 → R[0][0])   | 1.000 mA     | Encodes the intended MVM weight      |
| I_sneak (row1 → R[1][0])      | +0.200 mA    | Parasitic sneak path contribution    |
| **I_col0_actual**              | **1.200 mA** | What the sense amplifier measures    |
 
The sneak path introduces a **+20% error** relative to the ideal result (1.200 mA vs. 1.000 mA).
 
---
 
## (d) How Sneak Paths Corrupt MVM Results
 
In a resistive crossbar performing Matrix-Vector Multiplication (MVM), each column output current is supposed to represent exclusively the dot product of the input voltage vector with the programmed conductance weights of that column — i.e., only the cells connecting driven input rows to that column should contribute. However, when unselected rows and columns are left floating rather than actively driven to a fixed reference, they settle at intermediate voltages through resistive coupling (as shown above: V_row1 = 0.4 V, V_col1 = 0.6 V), creating parasitic current loops through off-state cells that inject additional current into the sensing column without any corresponding input stimulus. This error scales adversely with array size: in an N×N crossbar, each off-state cell can participate in O(N²) sneak paths, causing the accumulated parasitic current to grow with array dimensions and making large-scale analog MVM increasingly inaccurate unless mitigation strategies — such as selector devices (diodes, OTS selectors, or access transistors), active column clamping, or sneak-path-aware error correction — are employed.
