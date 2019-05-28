Release History
===============

1.1.0 ( in development)
------------

- Working to add support for MLP classification and Linear Regression.  
- Working to add support for model persistence once training has finished - including and transformation models.
- Working to add functionality to alert users when certain string-based categorical features should be encoded.  


1.0.5 (2019-05-27)
-------------------

**Bugfixes** 

- Fixed errors arising in testing because `self.objective` was replace with `self._objective` but not changed in the test suite.

**Improvements**

- Added coverage and links to shield.io badges on README.md


1.0.4 (2019-05-27)
-------------------

**Bugfixes** 

- Added conditions to catch the case where a search hasn't been executed on the data but user tries to print a summary
anyway

**Improvements**

...birth!