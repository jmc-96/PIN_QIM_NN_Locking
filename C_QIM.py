# =============================================================================
#                         QIM EMBEDING I EKSTRAKCIJA
# =============================================================================

# %% QIM EMBEDING

import numpy as np
  
def embed_qim(x, u, Δ=0.1):
    
    # Embed binary watermark u_bits into the cover vector x using QIM.
    # x : real cover signal
    # u_bits : binary watermark bits
    # Δ : float quantization step size
    # y : watermarked signal
    
    if len(u) > len(x):
        raise ValueError("Not enough cover elements for embedding all bits.")
    y = []
    for xi, ui in zip(x, u):
        if ui == 0:
            yi = Δ * np.round((xi - Δ / 4) / Δ) + Δ / 4
        else:
            yi = Δ * np.round((xi + Δ / 4) / Δ) - Δ / 4
        y.append(yi)
    
    y.extend(x[len(u):])
        
    return np.array(y)

# %% QIM EXTRACTION - Za len(y) = len(u)

def extract_qim(y, Δ=0.1):
    
    # u_hat : extracted binary watermark bits
    # Δ : float quantization step size
    # y : watermarked signal
    
    u_hat = []
    
    for yi in y:
        dist0 = abs(yi - (Δ * np.round((yi - Δ / 4) / Δ) + Δ / 4))
        dist1 = abs(yi - (Δ * np.round((yi + Δ / 4) / Δ) - Δ / 4))
        bit = 0 if dist0 < dist1 else 1
        u_hat.append(bit)
    
    return np.array(u_hat)

# %% DETEKCIJA DUZINE VODENOG ZIGA NA OSNOVU DECIMALNIH MESTA

def izdvoji_zigosan(y, bitska_duzina):
    def fourth_decimal_is_zero(val):
        # Get the 4th decimal digit
        val_str = f"{val:.10f}"  # Enough precision
        decimal_part = val_str.split('.')[-1]
        return len(decimal_part) >= 4 and decimal_part[3] == '0'

    f = 0
    for val in y:
        if fourth_decimal_is_zero(val):
            f += 1
        else:
            break

    # Adjust f da bi bio deljiv sa bitska_duzina
    f -= f % bitska_duzina
    # Return truncated list
    Y = y[:f]
    return Y


# %% QIM EXTRACTION - Za len(y) != len(u)

def extract_QIM(y, watermark_length, Δ=0.1):

    # Extract binary watermark bits from the first watermark_length elements of y using QIM.
    
    # y : watermarked signal
    # watermark_length : number of bits to extract
    # Δ : quantization step size
    # u_hat : extracted watermark bits

    u_hat = []

    for i in range(watermark_length):
        yi = y[i]
        dist0 = abs(yi - (Δ * np.round((yi - Δ / 4) / Δ) + Δ / 4))
        dist1 = abs(yi - (Δ * np.round((yi + Δ / 4) / Δ) - Δ / 4))
        bit = 0 if dist0 < dist1 else 1
        u_hat.append(bit)
    
    return np.array(u_hat)
