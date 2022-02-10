brutally simplified data reduction method comparisons

| Reduction Method | Pipeline | Compression Ratio | Benefits | Drawbacks | 
| ---              | ---      | ---               | --- | --- |
| Compression         | encode- > **bitstream storage** -> decode | high (TThresh) |
| Deep Super Resolution    | **LR+model storage** -> decode(SR) | low?
| Statistical Summary | **model storage** -> decode(sampling) | medium (beat SZ)
| importance sampling | **sample storage** -> decode(lerp, SR, ...) | ?
