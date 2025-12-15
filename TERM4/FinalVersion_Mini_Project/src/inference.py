import pandas as pd
from collections import OrderedDict

# Thresholds to top
ACF_OSC_THRESHOLD = 0.5
STICTION_SCORE_THRESHOLD = 0.5
OP_FLATNESS_THRESHOLD = 0.1
OP_JUMP_THRESHOLD = 0.01  # Normalized count
SPECTRAL_ENTROPY_NOISE = 2.0
ERROR_STD_HIGH = 1.0

SEVERITY_BANDS = {
    'LOW': (0.0, 0.33),
    'MEDIUM': (0.33, 0.66),
    'HIGH': (0.66, 1.01)
}

def severity_band(score):
    if score is None:
        return "NA"
    if score < 0.33:
        return "LOW"
    elif score < 0.66:
        return "MEDIUM"
    else:
        return "HIGH"

def generate_inferences(predictions, features=None, loop_id=None):
    """
    Generate human-readable inferences based on model predictions and key features.
    
    Args:
        predictions (dict): Dictionary of predicted classes/probabilities.
        features (dict): Dictionary of extracted features (optional).
        loop_id (str): ID of the loop.
        
    Returns:
        tuple: (OrderedDict(inferences), OrderedDict(debug_log))
    """
    inferences = OrderedDict()
    debug_log = OrderedDict()
    features = features or {}
    
    # 1. Normalize Predictions
    # This maps specific model keys to generic logical keys
    pred = {
        'osc_flag': predictions.get('Oscillation_Flag', 0),
        'osc_prob': predictions.get('Oscillation_Prob', None), # Assuming prob might be passed or inferred if available
        'noise_flag': predictions.get('Noise_Flag', 0),
        'noise_prob': predictions.get('Noise_Prob', None),
        'dist_flag': predictions.get('Disturbance_Flag', 0),
        'dist_prob': predictions.get('Disturbance_Prob', None),
        'stiction_score': predictions.get('Stiction_Score', 0)
    }
    
    # Note: If probabilities are not explicitly in 'predictions', we look for them or assume binary confidence
    # Check if 'Oscillation_Flag' is float (prob) or strictly binary.
    # But usually predictions dict comes from models.py which gives values or classes.
    # The current models.py gives 'Oscillation_Flag' as 0/1 (class).
    
    # --- 1. Oscillation Inference ---
    is_oscillating = pred['osc_flag'] == 1
    
    if is_oscillating:
        freq = features.get('psd_dom_freq', 0)
        period = 1/freq if freq > 0 else 0
        
        conf_str = ""
        if pred['osc_prob'] is not None:
             conf_str = f" (confidence: {pred['osc_prob']:.2f})"

        # Severity
        osc_sev = features.get('oscillation_severity', 0)
        osc_band = severity_band(osc_sev)
        
        inferences['Oscillation'] = f"Oscillation detected{conf_str} (Severity: {osc_band}, Score: {osc_sev:.2f}) with dominant frequency {freq:.3f} Hz."
        
        debug_log['Oscillation'] = {
            'condition': 'osc_flag == 1',
            'value': pred['osc_flag'],
            'prob': pred['osc_prob'],
            'psd_dom_freq': freq,
            'severity': osc_sev,
            'acf_peak': features.get('acf_peak_val')
        }
    else:
        # Near-miss / Borderline
        # If prob is available and high-ish
        if pred['osc_prob'] and pred['osc_prob'] > 0.4:
            inferences['Oscillation_Watch'] = f"Loop shows borderline oscillatory behavior (confidence: {pred['osc_prob']:.2f})."
            debug_log['Oscillation'] = {'condition': 'prob > 0.4 (Borderline)', 'prob': pred['osc_prob']}
        # Check feature evidence even if model says No
        elif features.get('acf_peak_val', 0) > ACF_OSC_THRESHOLD:
             inferences['Oscillation_Risk'] = "No strong oscillation predicted, but some periodicity observed in ACF."
             debug_log['Oscillation'] = {'condition': f'acf_peak > {ACF_OSC_THRESHOLD}', 'acf_peak': features.get('acf_peak_val')}


    # --- 2. Stiction Inference ---
    stiction_score = pred['stiction_score']
    
    if isinstance(stiction_score, (int, float)): 
        if stiction_score > STICTION_SCORE_THRESHOLD:
            # Severity for Stiction is the score itself basically
            stic_band = severity_band(stiction_score)
            score_norm = stiction_score
            if score_norm > 1.0: score_norm /= 100.0
            stic_band = severity_band(score_norm)

            inferences['Stiction'] = f"High probability of valve stiction ({stiction_score:.2f}) - Severity: {stic_band}."
            
            # Evidence Evidence
            causes = []
            flat_val = features.get('op_flatness_ratio', 0)
            if flat_val > OP_FLATNESS_THRESHOLD:
                msg = f"Evidence: Controller output is flat for {flat_val*100:.1f}% of the time."
                inferences['Stiction_Evidence_Flatness'] = msg
                causes.append({'factor': 'flatness', 'val': flat_val})
            
            jump_val = features.get('op_jump_count', 0)
            if jump_val > OP_JUMP_THRESHOLD:
                 msg = f"Evidence: Frequent step changes in OP detected ({jump_val*100:.1f}% samples)."
                 inferences['Stiction_Evidence_Jumps'] = msg
                 causes.append({'factor': 'jumps', 'val': jump_val})
            
            debug_log['Stiction'] = {
                'condition': f'score > {STICTION_SCORE_THRESHOLD}', 
                'score': stiction_score,
                'evidence': causes
            }
    
    # --- 3. Noise Inference ---
    if pred['noise_flag'] == 1:
        # Severity
        noise_sev = features.get('noise_severity', 0)
        noise_band = severity_band(noise_sev)

        inferences['Noise'] = f"Signal is classified as Noisy (Severity: {noise_band}, Score: {noise_sev:.2f})."
        
        if features.get('spectral_entropy', 0) > SPECTRAL_ENTROPY_NOISE:
            inferences['Noise_Detail'] = "High spectral entropy indicates random variation."
        
        debug_log['Noise'] = {
             'condition': 'noise_flag == 1', 
             'entropy': features.get('spectral_entropy'),
             'severity': noise_sev
        }
            
    # --- 4. Disturbance Inference ---
    if pred['dist_flag'] == 1:
        dist_sev = features.get('disturbance_severity', 0)
        dist_band = severity_band(dist_sev)
        inferences['Disturbance'] = f"External disturbance detected (Severity: {dist_band}, Score: {dist_sev:.2f})."
        debug_log['Disturbance'] = {'condition': 'dist_flag == 1', 'severity': dist_sev}
        
    # --- 5. Performance/Variance ---
    error_std = features.get('error_std', 0)
    if error_std > ERROR_STD_HIGH:
        inferences['Performance'] = f"High control error variance (Std Dev: {error_std:.2f}). Loop performance may be poor."
        debug_log['Performance'] = {'condition': f'error_std > {ERROR_STD_HIGH}', 'val': error_std}
        
    if not inferences:
        inferences['Status'] = "Loop appears stable with no significant anomalies detected."
        
    return inferences, debug_log
