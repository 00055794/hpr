"""
Streamlit Web Application for KZ Real Estate Price Prediction
Uses the complete pipeline that exactly matches notebook training.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
try:
    from folium.plugins import MousePosition, Geocoder
except ImportError:
    from folium.plugins import MousePosition
    Geocoder = None
import traceback
from datetime import datetime
import requests
import base64

# Import our notebook-matched pipeline
from pipeline_complete import CompletePipeline

st.set_page_config(page_title="KZ Real Estate Price Estimator", layout="wide")

# Load and encode background image
@st.cache_data
def get_background_image():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    bg_path = os.path.join(app_dir, "background.jpg")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{data}"
    return None

bg_image = get_background_image()

if bg_image:
    st.markdown(f"""
    <style>
        .stApp {{
            background-image: url('{bg_image}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(14, 17, 23, 0.85);
            z-index: -1;
        }}
        /* Frame styling for columns */
        [data-testid="column"] {{
            background-color: rgba(50, 50, 50, 0.5);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
    </style>
    """, unsafe_allow_html=True)

st.title("KZ Real Estate Price Estimator")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the complete prediction pipeline once"""
    try:
        # Get the directory where this script is located
        app_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(app_dir)
        
        pipeline = CompletePipeline(
            model_dir=os.path.join(app_dir, "nn_model"),
            region_grid_lookup=os.path.join(app_dir, "region_grid_lookup.json"),
            region_grid_encoder=os.path.join(app_dir, "region_grid_encoder.json"),
            segments_geojson=os.path.join(parent_dir, "segments_fine_heuristic_polygons.geojson")
        )
        return pipeline, None
    except Exception as e:
        return None, f"Failed to load pipeline: {str(e)}\n{traceback.format_exc()}"


# Load pipeline
pipeline, error = load_pipeline()

if error:
    st.error(f"Pipeline Load Error: {error}")
    st.stop()

# Two-column layout: Single Prediction | Batch Upload
col_single, col_batch = st.columns([1, 1])

# ==================== LEFT: Single Prediction ====================
with col_single:
    st.subheader("Single Prediction")
    
    # Map picker (collapsed by default)
    with st.expander("Pick location on map or Search by Address", expanded=False):
        # Use session state for coordinates
        if "LATITUDE" not in st.session_state:
            st.session_state["LATITUDE"] = 43.2567
        if "LONGITUDE" not in st.session_state:
            st.session_state["LONGITUDE"] = 76.9286
        
        pick_lat = st.session_state.get("LATITUDE", 43.2567)
        pick_lon = st.session_state.get("LONGITUDE", 76.9286)
        
        m = folium.Map(location=[pick_lat, pick_lon], zoom_start=12, tiles=None)
        
        # Add satellite and OSM layers
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
            overlay=False,
            control=True,
            show=True
        ).add_to(m)
        folium.TileLayer("OpenStreetMap", name="OSM", overlay=False, control=True, show=False).add_to(m)
        
        # Add mouse position and click popup
        MousePosition(position="topright", prefix="Lat/Lon:", separator=", ", num_digits=6).add_to(m)
        folium.LatLngPopup().add_to(m)
        
        # Add geocoder search if available (compact, near zoom controls)
        if Geocoder is not None:
            try:
                Geocoder(
                    collapsed=True,
                    position="topleft",
                    placeholder="Search...",
                    add_marker=True
                ).add_to(m)
            except Exception:
                pass
        
        # Add marker at current location
        folium.Marker(
            [pick_lat, pick_lon],
            popup=f"{pick_lat:.6f}, {pick_lon:.6f}",
            tooltip="Current location"
        ).add_to(m)
        
        folium.LayerControl(collapsed=True).add_to(m)
        
        # Display map and capture clicks
        map_data = st_folium(m, height=350, use_container_width=True, key="map_picker")
        
        # Update coordinates from map click
        if map_data and map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            if clicked_lat and clicked_lon:
                st.session_state["LATITUDE"] = clicked_lat
                st.session_state["LONGITUDE"] = clicked_lon
                st.success(f"Selected: {clicked_lat:.6f}, {clicked_lon:.6f}")
    
    # Compact form layout - 3 rows with multiple columns
    # Row 1: Location and Size
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    LATITUDE = r1c1.number_input("Latitude", 40.0, 55.0, st.session_state.get("LATITUDE", 43.2567), format="%.6f")
    LONGITUDE = r1c2.number_input("Longitude", 46.0, 87.0, st.session_state.get("LONGITUDE", 76.9286), format="%.6f")
    ROOMS = r1c3.number_input("Rooms", 1, 10, 2)
    TOTAL_AREA = r1c4.number_input("Area (m²)", 10.0, 500.0, 62.0, 1.0)
    
    # Row 2: Building details (5 columns to fit Floor/Total separately)
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
    FLOOR = r2c1.number_input("Floor", 1, 100, 5)
    TOTAL_FLOORS = r2c2.number_input("Total floors", 1, 100, 9)
    YEAR = r2c3.number_input("Year", 1950, 2025, 2015)
    CEILING = r2c4.number_input("Ceiling (m)", 2.0, 5.0, 2.7, 0.1)
    MATERIAL = r2c5.selectbox("Material", [1, 2, 3, 4], 1, format_func=lambda x: {1: "Brick", 2: "Panel", 3: "Monolith", 4: "Other"}[x])
    
    # Row 3: Quality
    r3c1, r3c2 = st.columns(2)
    FURNITURE = r3c1.selectbox("Furniture", [1, 2, 3], 1, format_func=lambda x: {1: "No", 2: "Partial", 3: "Full"}[x])
    CONDITION = r3c2.selectbox("Condition", [1, 2, 3, 4, 5], 2, format_func=lambda x: f"{x} - {'Poor' if x==1 else 'Fair' if x==2 else 'Good' if x==3 else 'Excellent' if x==4 else 'Perfect'}")
    
    # Predict button
    if st.button("Predict Price", use_container_width=True):
        try:
            # Prepare input
            input_data = {
                'ROOMS': ROOMS,
                'LONGITUDE': LONGITUDE,
                'LATITUDE': LATITUDE,
                'TOTAL_AREA': TOTAL_AREA,
                'FLOOR': FLOOR,
                'TOTAL_FLOORS': TOTAL_FLOORS,
                'FURNITURE': FURNITURE,
                'CONDITION': CONDITION,
                'CEILING': CEILING,
                'MATERIAL': MATERIAL,
                'YEAR': YEAR
            }
            
            # Make prediction
            price_kzt, features_df = pipeline.predict_single(input_data, return_features=True)
            
            # Display result
            st.success(f"Predicted Price: {price_kzt:,.0f} KZT")
            st.metric("Price per m²", f"{price_kzt/TOTAL_AREA:,.0f} KZT/m²")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ==================== RIGHT: Batch Upload ====================
with col_batch:
    st.subheader("Batch Upload")
    
    upload = st.file_uploader("Upload CSV file", type=["csv"], help="CSV with required columns")
    
    # Download previous predictions
    b_csv = st.session_state.get("batch_csv_bytes")
    b_csv_name = st.session_state.get("batch_csv_name", "predictions.csv")
    
    if b_csv:
        st.download_button(
            label="Download predictions",
            data=b_csv,
            file_name=b_csv_name,
            mime="text/csv",
            use_container_width=True
        )
    
    if upload is not None:
        try:
            # Read CSV with full float precision for lat/lon
            df_in = pd.read_csv(upload, float_precision='high')
            
            # Ensure lat/lon have full precision (not rounded)
            if 'LATITUDE' in df_in.columns:
                df_in['LATITUDE'] = df_in['LATITUDE'].astype(float)
            if 'LONGITUDE' in df_in.columns:
                df_in['LONGITUDE'] = df_in['LONGITUDE'].astype(float)
            
            # Make predictions
            predictions_kzt = pipeline.predict_batch(df_in)
            
            # Add predictions to dataframe
            df_out = df_in.copy()
            df_out['pred_price_kzt'] = predictions_kzt
            
            st.success(f"Predicted {len(df_out)} properties")
            
            # Show results with all input features + prediction
            display_cols = [c for c in ["ROOMS", "LONGITUDE", "LATITUDE", "TOTAL_AREA", "FLOOR", "TOTAL_FLOORS", "FURNITURE", "CONDITION", "CEILING", "MATERIAL", "YEAR", "pred_price_kzt"] if c in df_out.columns]
            
            # Format display to show full lat/lon precision
            df_display = df_out[display_cols].head(50).copy()
            st.dataframe(df_display, use_container_width=True)
            
            # Prepare download with full precision
            import time
            csv_bytes = df_out.to_csv(index=False, float_format='%.10f').encode("utf-8")
            st.session_state["batch_csv_bytes"] = csv_bytes
            _ts = time.strftime("%Y%m%d_%H%M%S")
            st.session_state["batch_csv_name"] = f"predictions_{_ts}.csv"
            
        except Exception as e:
            st.error(f"Batch prediction failed: {str(e)}")
    
    # Template download
    with st.expander("Download template"):
        template_data = pd.DataFrame([{
            'ROOMS': 2,
            'LONGITUDE': 76.9286,
            'LATITUDE': 43.2567,
            'TOTAL_AREA': 62.0,
            'FLOOR': 5,
            'TOTAL_FLOORS': 9,
            'FURNITURE': 2,
            'CONDITION': 3,
            'CEILING': 2.7,
            'MATERIAL': 2,
            'YEAR': 2015
        }])
        
        csv_template = template_data.to_csv(index=False)
        st.download_button(
            label="Download CSV template",
            data=csv_template,
            file_name="template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # ==================== COMMENTS SECTION (Ultra Compact) ====================
    st.markdown("---")
    st.markdown("**User Feedback**")
    
    with st.form("comment_form", clear_on_submit=True):
        fcol1, fcol2, fcol3 = st.columns([2, 3, 1])
        with fcol1:
            user_email = st.text_input("Email", placeholder="your@email.com", label_visibility="collapsed")
        with fcol2:
            user_comment = st.text_input("Comment", placeholder="Share your feedback...", label_visibility="collapsed")
        with fcol3:
            submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            if user_comment.strip():
                try:
                    # Prepare comment data
                    comment_data = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'email': user_email.strip() if user_email else '',
                        'comment': user_comment.strip()
                    }
                    
                    # GitHub API configuration
                    github_token = st.secrets.get("GITHUB_TOKEN", "")
                    repo_owner = "00055794"
                    repo_name = "hpr"
                    file_path = "comments.csv"
                    
                    if github_token:
                        # GitHub API endpoint
                        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
                        headers = {
                            "Authorization": f"token {github_token}",
                            "Accept": "application/vnd.github.v3+json"
                        }
                        
                        # Get existing file
                        response = requests.get(api_url, headers=headers)
                        
                        if response.status_code == 200:
                            # File exists - update it
                            file_info = response.json()
                            existing_content = base64.b64decode(file_info['content']).decode('utf-8')
                            df_comments = pd.read_csv(pd.io.common.StringIO(existing_content))
                            df_comments = pd.concat([df_comments, pd.DataFrame([comment_data])], ignore_index=True)
                            sha = file_info['sha']
                        else:
                            # File doesn't exist - create new
                            df_comments = pd.DataFrame([comment_data])
                            sha = None
                        
                        # Convert to CSV and encode
                        csv_content = df_comments.to_csv(index=False)
                        encoded_content = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
                        
                        # Prepare commit data
                        commit_data = {
                            "message": f"Add user comment from {comment_data['email'] or 'anonymous'}",
                            "content": encoded_content,
                            "branch": "main"
                        }
                        if sha:
                            commit_data["sha"] = sha
                        
                        # Push to GitHub
                        push_response = requests.put(api_url, json=commit_data, headers=headers)
                        
                        if push_response.status_code in [200, 201]:
                            st.success("Submitted")
                        else:
                            st.error(f"Failed to save: {push_response.status_code}")
                    else:
                        # Fallback to local file if no GitHub token
                        app_dir = os.path.dirname(os.path.abspath(__file__))
                        comments_file = os.path.join(app_dir, 'comments.csv')
                        
                        if os.path.exists(comments_file):
                            df_comments = pd.read_csv(comments_file)
                            df_comments = pd.concat([df_comments, pd.DataFrame([comment_data])], ignore_index=True)
                        else:
                            df_comments = pd.DataFrame([comment_data])
                        
                        df_comments.to_csv(comments_file, index=False)
                        st.success("Submitted")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a comment.")

