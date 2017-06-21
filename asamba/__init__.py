# Copyright 2017 Ehsan Moravveji (KU Leuven)
#
# Licensed under GPL, so you can only use this package if you comply with the terms
# under this license.
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__all__       = ('__author__', '__copyright__', '__email__', '__license__', '__scope__',
                 '__summary__', '__title__', '__url__', '__version__')

__title__     = 'asamba'
__version__   = '1.0.8'
__summary__   = 'Python interface to peform asteroseismic modelling with the asamba grid'
# __url__       = 'https://pypi.python.org/pypi/asamba'
# __url__       = 'git@github.com:moravveji/asamba.git'
__url__       = 'https://fys.kuleuven.be/ster/Projects/ASAMBA'
__author__    = 'Ehsan Moravveji, KU Leuven, Belgium'
__email__     = 'Ehsan.Moravveji@kuleuven.be'
__license__   = 'GPL'
__copyright__ = 'Copyright 2017 Ehsan Moravveji'
__scope__     = 'ASAMBA (AsteroSeismic Approach towards understanding Massive Blue stArs) is a Marie Curie project \
                that tries to infer deep physical understanding of the internal structure and evolution of massive \
                stars in the light of recent very high precision space observations of pulsating massive stars.\n\n \
                Under this umbrella, a large grid of stellar models (using MESA) are computed, and the theoretical \
                pulsation frequencies of each model (after iterating over various rotation rates) are also computed\
                using the (GYRE) code. From this rich dataset (~3.8 million stellar models, and 42 million frequency\
                lists), a PostgreSQL database is built, and is made openly accessible.\n\n \
                The present Python package provides a convenient user interface that offers all available functionalities \
                to the users, and allows them to interact with the database, and conduct their own research of interest. \
                Needless to say that the user must have a full understanding of the meaning of the parameters he/she uses \
                which steer the analysis. We strongly recommend reading the source code, the documentation around most of \
                the code blocks, and the compiled documentation pages that ships in with this package.'
