Release History
===============

dev
---

1.0.4 (2019-05-27)
-------------------

**Bugfixes** 

- Added conditions to catch the case where a search hasn't been executed on the data but user tries to print a summary
anyway

**Improvements**

- Added relative import so that __MLAutomator__ class can be imported with: 

```Python
import mlautomator
```
rather than 
```Python
from mlautomator.mlautomator import MLAutomator
```