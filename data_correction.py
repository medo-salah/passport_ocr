import streamlit as st
import logging

logger = logging.getLogger(__name__)

def setup_manual_correction(data, context="default"):
    """
    Setup UI for manual correction of extracted data

    Args:
        data: Dictionary containing extracted data
        context: Context string to ensure unique widget keys

    Returns:
        corrected_data: Dictionary with corrected data
        was_corrected: Boolean indicating if any corrections were made
    """
    corrected_data = {}
    was_corrected = False

    # Initialize session state for form data if not exists
    form_key = f"correction_form_data_{context}"
    if form_key not in st.session_state:
        st.session_state[form_key] = {}

    # Create a form for manual corrections
    with st.form(key=f"correction_form_{context}", clear_on_submit=False):
        st.write("Edit any incorrect fields below:")

        # Create input fields for each data item
        for key, value in data.items():
            # Skip internal validation data
            if key == "_validation":
                continue

            # Determine if field has validation issues
            has_issue = False
            if "_validation" in data and key in data["_validation"]:
                has_issue = not data["_validation"][key]["valid"]

            # Add visual indicator for fields with issues
            prefix = "⚠️ " if has_issue else ""

            # Get the current value (either from session state or original data)
            current_value = st.session_state[form_key].get(key, value if value != "Not Found" else "")

            # Create text input with unique key
            corrected_value = st.text_input(
                f"{prefix}{key}",
                value=current_value,
                key=f"correction_{key}_{context}",
                help=f"Original value: {value}" if value != "Not Found" else "No original value found"
            )

            # Store corrected value
            corrected_data[key] = corrected_value
            st.session_state[form_key][key] = corrected_value

            # Check if value was changed
            if corrected_value != value and corrected_value != "" and value != "Not Found":
                was_corrected = True

        # Create columns for buttons
        col1, col2 = st.columns(2)

        with col1:
            # Submit button
            submit_button = st.form_submit_button(
                "Apply Corrections",
                help="Click to apply the corrections to your data",
                type="primary"
            )

        with col2:
            # Reset button
            reset_button = st.form_submit_button(
                "Reset Form",
                help="Reset all fields to original values"
            )

        if submit_button:
            st.success("✅ Corrections applied successfully")
            was_corrected = True

            # Log the correction event
            try:
                from analytics_dashboard import log_analytics_event
                log_analytics_event("manual_correction_applied", {
                    "fields_corrected": len([k for k, v in corrected_data.items() if v != data.get(k, "")]),
                    "context": context
                })
            except ImportError:
                pass  # Analytics not available

        if reset_button:
            # Clear the form data from session state
            st.session_state[form_key] = {}
            st.info("🔄 Form reset to original values")
            st.rerun()

    return corrected_data, was_corrected

def apply_corrections(original_data, corrected_data):
    """
    Apply corrections to the original data and update validation status
    
    Args:
        original_data: Original data dictionary
        corrected_data: Corrected data dictionary
        
    Returns:
        updated_data: Data with corrections and updated validation
    """
    # Create a copy of the original data
    updated_data = original_data.copy()
    
    # Apply corrections
    for key, value in corrected_data.items():
        if key in updated_data and updated_data[key] != value:
            updated_data[key] = value
    
    # Update validation results if present
    if "_validation" in updated_data:
        validation = updated_data["_validation"]
        
        # For each corrected field, update its validation status
        for key in validation.keys():
            if key in corrected_data and original_data.get(key) != corrected_data.get(key):
                # Mark manually corrected fields as valid
                validation[key] = {
                    "valid": True,
                    "message": "Manually corrected"
                }
        
        updated_data["_validation"] = validation
    
    return updated_data
