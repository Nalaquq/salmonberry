We calculate a novel terrain metric labeled as "flank-persistence." This new method was named Gyabaah's method. We used the gradient components $z_x$ and $z_y$ and the magnitude $g_0=\sqrt{z_x^2+z_y^2}$ caluclated by the CDM to compute flank-persistence.

For each focal cell $p$ where $g_0$ is finite:

$$u=(z_x,z_y)/g_0,\qquad p_\pm = p \pm d\,u$$

$$
g_\pm = \text{bilinear sample of } g \text{ at } p_\pm,\qquad
\boxed{F_d=\min\!\left(\frac{g_+}{g_0},\frac{g_-}{g_0}\right)}
$$

The samples are straight-line offsets along the focal gradient direction. They do not follow
the terrain. Only $F_d$ is written; $g_+$ and $g_-$ stay internal.

The table below explains the interpretation of this custom metric.

`Note: $F_d$ is the weaker of the two directional persistence ratios.`

| Value of $F_d$  | Meaning                                                                                                                                  |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| $F_d \approx 1$ | The gradient magnitude a distance $d$ uphill and downhill is about the same as at the focal cell, for example on a uniform planar flank. |
| $F_d < 1$       | The gradient weakens on at least one side, for example near a crest, toe, bench or break of slope.                                       |
| $F_d > 1$       | Both sides are steeper than the focal cell, for example in a local gradient minimum within a slope. The value is not clipped.            |
