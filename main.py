import streamlit as st
import numpy as np
from pathlib import Path
import traceback
import tifffile
import cv2

# Import local script functions
from cellpose_batch_segmentation import process_image, get_all_tiff_files, save_masks

# Page setup: wide layout
st.set_page_config(layout="wide")
st.title("Cell Segmentation Editor")

# Initialize session state (if needed)
if 'image_files' not in st.session_state:
    st.session_state.image_files = []       # Holds image paths
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0      # Index of current image
if 'reset_folder_input' not in st.session_state:
    st.session_state.reset_folder_input = False # Flag to clear folder input
if 'show_save_all_summary' not in st.session_state:
    st.session_state.show_save_all_summary = False # Flag to show batch save summary
if 'save_all_summary_message' not in st.session_state:
    st.session_state.save_all_summary_message = "" # Batch save result message
if 'save_all_errors' not in st.session_state:
    st.session_state.save_all_errors = [] # Batch save error list

# Handle request to clear folder input
if st.session_state.reset_folder_input:
    if 'folder_path_input' in st.session_state:
        st.session_state.folder_path_input = ""
    st.session_state.reset_folder_input = False # Reset flag

# Display batch save summary (if flagged)
if st.session_state.show_save_all_summary:
    st.success(st.session_state.save_all_summary_message) # Show summary message 
    if st.session_state.save_all_errors: # Show errors if any
        with st.expander("See batch processing errors"):
            for error_msg in st.session_state.save_all_errors:
                st.warning(error_msg)
    # Reset summary display state
    st.session_state.show_save_all_summary = False
    st.session_state.save_all_summary_message = ""
    st.session_state.save_all_errors = []

# Setup two layout columns
col1, col2 = st.columns([1, 2]) 


with col1:
    # Get folder path input
    patient_folder_path = st.text_input(
        "Enter patient folder",
        placeholder="Type something...",
        key='folder_path_input' 
    )

    # Load images if path given & list empty
    if patient_folder_path and ('image_files' not in st.session_state or not st.session_state.image_files):
        try:
            st.session_state.image_files = get_all_tiff_files(patient_folder_path)
            st.session_state.current_index = 0 # Reset index
            if not st.session_state.image_files:
                st.warning("No images found.") # Show warning
        except Exception as e:
            st.error(f"Error loading images: {e}") # Show error
            # st.code(traceback.format_exc()) # For debug

    # Image selection dropdown
    selected_image_str = None
    if st.session_state.image_files: # Only if images are loaded
        # Ensure index is valid
        if st.session_state.current_index >= len(st.session_state.image_files):
            st.session_state.current_index = 0

        image_options = [str(f) for f in st.session_state.image_files]
        # Selectbox linked to session state index
        st.session_state.current_index = st.selectbox(
            "Select an image",
            options=range(len(image_options)), # Options are indices
            format_func=lambda i: image_options[i], # Display filenames
            index=st.session_state.current_index,
            key='image_selector'
        )
        # Get selected image path string
        selected_image_str = image_options[st.session_state.current_index]

    # Cellpose parameter sliders
    st.markdown("**Cellpose Parameters**")
    diameter = st.slider("Diameter", 1, 50, 25)
    cell_prob_threshold = st.slider("Cell Probability Threshold", 0.0, 1.0, 0.5, 0.05)
    flow_threshold = st.slider("Flow Threshold", 0.0, 1.0, 0.5, 0.05)
    dilation = st.slider("Cytoplasm Dilation", 1, 15, 7, 1)

    # Action buttons
    save_button = st.button("Save Mask")
    save_all_button = st.button("Save All Masks")

    # --- Save All Logic ---
    if save_all_button:
        if st.session_state.image_files: # Check images exist
            # Get current parameters
            current_diameter = diameter
            current_cell_prob = cell_prob_threshold
            current_flow_thresh = flow_threshold
            current_dilation = dilation

            total_files = len(st.session_state.image_files)
            # Placeholders for progress display
            progress_bar_placeholder = st.empty()
            status_text_placeholder = st.empty()

            saved_count = 0
            skipped_count = 0
            error_list = []

            # Process all loaded images
            for idx, img_path in enumerate(list(st.session_state.image_files)):
                img_path_str = str(img_path)
                # Update progress bar/text
                progress_text = f"Processing {idx+1}/{total_files}: {img_path.name}..."
                status_text_placeholder.text(progress_text)
                progress_bar_placeholder.progress((idx + 1) / total_files)

                try:
                    # Process one image
                    _, batch_cytoplasm_masks = process_image(
                        img_path_str,
                        diameter=current_diameter,
                        cell_prob_threshold=current_cell_prob,
                        flow_threshold=current_flow_thresh,
                        dilation=current_dilation
                    )
                    # Save colored mask if cells found
                    if batch_cytoplasm_masks is not None:
                        # Save the grayscale mask with unique IDs (not the color display)
                        save_masks(img_path_str, batch_cytoplasm_masks) 
                        saved_count += 1
                    else:
                        skipped_count += 1 # No cells detected
                except Exception as e:
                    error_list.append(f"{img_path.name}: {e}") # Record error
                    skipped_count += 1

            # Clear progress display
            progress_bar_placeholder.empty()
            status_text_placeholder.empty()

            # Store results for display after rerun
            st.session_state.save_all_summary_message = f"Batch complete! Saved {saved_count}, Skipped {skipped_count}."
            st.session_state.save_all_errors = list(error_list)
            st.session_state.show_save_all_summary = True # Signal display

            # Reset app state
            st.session_state.pop('image_files', None)
            st.session_state.current_index = 0
            st.session_state.reset_folder_input = True
            st.rerun() # Rerun to apply reset
        else:
            st.warning("No images loaded to save.")

# --- Output Column (col2) ---
with col2:
    # Only if an image is selected
    if selected_image_str:
        try:
            # Process current image
            with st.spinner("Processing image..."):
                img, cytoplasm_masks = process_image(
                    selected_image_str,
                    diameter=diameter,
                    cell_prob_threshold=cell_prob_threshold,
                    flow_threshold=flow_threshold,
                    dilation=dilation
                )

            # Display overlay if valid
            if cytoplasm_masks is not None and img is not None:
                # Normalize & convert original image
                if img.dtype != np.uint8:
                     img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Create colored mask
                mask_display = np.zeros_like(img_bgr, dtype=np.uint8)
                for i in range(1, cytoplasm_masks.max() + 1):
                    mask_display[cytoplasm_masks == i] = np.random.randint(50, 255, 3, dtype=np.uint8)

                # Blend original and mask
                alpha = 0.6 # Original image visibility
                beta = 0.4  # Mask visibility
                blended_image = cv2.addWeighted(img_bgr, alpha, mask_display, beta, 0)

                # Save button action (single image)
                if save_button:
                    with st.spinner("Saving mask..."):
                        # Save the grayscale mask with unique IDs
                        output_path = save_masks(selected_image_str, cytoplasm_masks) 
                        st.success(f"Mask saved: {output_path}")

                    # Remove image & advance/reset
                    if st.session_state.image_files:
                        current_idx_to_remove = st.session_state.current_index
                        st.session_state.image_files.pop(current_idx_to_remove)

                        if not st.session_state.image_files: # List now empty?
                            st.session_state.current_index = 0
                            st.session_state.pop('image_files')
                            st.session_state.reset_folder_input = True
                            st.toast("All images processed!")
                        else: # List not empty
                            if st.session_state.current_index >= len(st.session_state.image_files):
                                st.session_state.current_index = len(st.session_state.image_files) - 1
                        st.rerun() # Update UI

                # Display the blended overlay
                st.image(blended_image, caption="Segmentation Overlay Preview", use_container_width=True)

            elif img is None: # Handle missing original image
                 st.warning("Could not load original image.")
            else: # Handle no cells detected
                st.warning("No cells were detected.")
        except Exception as e:
            st.error(f"Error processing image: {e}")
            # st.code(traceback.format_exc())

    # Handle states when no image is selected
    elif patient_folder_path:
        if not st.session_state.image_files: # Folder processed or empty
             st.success("🎉 Ready for new folder.")
    else:
        st.info("⬆️ Enter patient folder path.")





