import ssl

import graphein

ssl._create_default_https_context = ssl._create_unverified_context

graphein.verbose(False)
