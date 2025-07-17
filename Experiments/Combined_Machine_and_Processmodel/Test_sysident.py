import numpy as np
import sysidentpy as sip
from sysidentpy.model_structure_selection import NARMAX
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.generate_data import get_miso_data

# Schritt 2: Daten vorbereiten
# Angenommen, Sie haben Ihre Daten als numpy-Arrays
input_data = np.random.rand(1000, 2)  # Beispiel-Eingabedaten
output_data = np.random.rand(1000)  # Beispiel-Ausgabedaten



# Modellvorhersage
yhat = model.predict(X=input_data, steps_ahead=10)

# Modellgüte bewerten
rrse = root_relative_squared_error(output_data, yhat)
print(f"Root Relative Squared Error: {rrse}")
