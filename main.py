import streamlit as st
import tensorflow as tf
import numpy as np
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("AI Plant Detection")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE DETECTION", "CROP VIABILITY GUIDE", "FARMING GUIDE","ABOUT THE PROJECT", "ABOUT US"])
#app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# import Image from pillow to open images
from PIL import Image
img = Image.open("Diseases.png")

# display image using streamlit
st.image(img)

#Main Page
if(app_mode=="HOME"):
        # Homepage UI
    st.markdown("""
        <h1 style='text-align: center; color: green;'>🌿 Plant Disease Detection 🌿</h1>
        <p style='text-align: center; font-size: 18px;'>Harness the power of AI to diagnose plant diseases and ensure healthier crops.</p>
        <hr>
    """, unsafe_allow_html=True)

    # About Section
    st.markdown("""
    ### 🌱 About This App
    This application helps farmers and agricultural experts detect plant diseases with the help of AI-powered image processing. 
    Simply upload a picture of a leaf, and our model will analyze and predict potential diseases.

    ### 🔍 How It Works
    1. **Capture or Upload**: Take a clear picture of the affected plant.
    2. **Analyze**: The AI model processes the image and identifies possible diseases.
    3. **Get Results**: Receive an instant diagnosis with suggestions for treatment.

    ### 🚀 Get Started
    Use the sidebar to navigate and start detecting plant diseases!
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <hr>
        <p style='text-align: center;'>© 2025 Plant Health AI | Powered by Machine Learning & Computer Vision</p>
    """, unsafe_allow_html=True)


# CROP VIABILITY GUIDE Page 
elif app_mode == "CROP VIABILITY GUIDE":
    st.markdown("""
        <h1 style='text-align: center; color: green;'>🌿 CROP VIABILITY GUIDE 🌿</h1>
    """, unsafe_allow_html=True)


    cropData = [
        {"name": "Apple", "nitrogen": 20.80, "phosphorus": 134.22, "potassium": 199.89, "temperature": 22.63, "humidity": 92.33, "pH": 5.93, "rainfall": 112.65},
        {"name": "Banana", "nitrogen": 100.23, "phosphorus": 82.01, "potassium": 50.05, "temperature": 27.38, "humidity": 80.36, "pH": 5.98, "rainfall": 104.63},
        {"name": "Blackgram", "nitrogen": 40.02, "phosphorus": 67.47, "potassium": 19.24, "temperature": 29.97, "humidity": 65.12, "pH": 7.13, "rainfall": 67.88},
        {"name": "Chickpea", "nitrogen": 40.09, "phosphorus": 67.79, "potassium": 79.92, "temperature": 18.87, "humidity": 16.86, "pH": 7.34, "rainfall": 80.06},
        {"name": "Coconut", "nitrogen": 21.98, "phosphorus": 16.93, "potassium": 30.59, "temperature": 27.41, "humidity": 94.84, "pH": 5.98, "rainfall": 175.69},
        {"name": "Coffee", "nitrogen": 101.20, "phosphorus": 28.74, "potassium": 29.94, "temperature": 25.54, "humidity": 58.87, "pH": 6.81, "rainfall": 158.07},
        {"name": "Cotton", "nitrogen": 117.77, "phosphorus": 46.24, "potassium": 19.56, "temperature": 23.99, "humidity": 79.84, "pH": 6.92, "rainfall": 80.09},
        {"name": "Grapes", "nitrogen": 23.18, "phosphorus": 132.53, "potassium": 200.11, "temperature": 23.87, "humidity": 81.87, "pH": 6.25, "rainfall": 69.91},
        {"name": "Jute", "nitrogen": 78.40, "phosphorus": 46.86, "potassium": 39.99, "temperature": 24.96, "humidity": 79.64, "pH": 6.73, "rainfall": 174.79},
        {"name": "Lentil", "nitrogen": 18.77, "phosphorus": 68.36, "potassium": 19.41, "temperature": 24.51, "humidity": 64.80, "pH": 6.99, "rainfall": 45.68},
        {"name": "Maize", "nitrogen": 77.76, "phosphorus": 48.44, "potassium": 19.79, "temperature": 22.61, "humidity": 65.92, "pH": 6.26, "rainfall": 84.76},
        {"name": "Mango", "nitrogen": 20.07, "phosphorus": 27.18, "potassium": 29.92, "temperature": 31.90, "humidity": 50.05, "pH": 5.77, "rainfall": 94.99},
        {"name": "Mothbeans", "nitrogen": 21.44, "phosphorus": 48.01, "potassium": 20.23, "temperature": 28.52, "humidity": 53.16, "pH": 6.85, "rainfall": 51.22},
        {"name": "Mungbean", "nitrogen": 20.99, "phosphorus": 47.28, "potassium": 19.87, "temperature": 28.27, "humidity": 85.95, "pH": 6.74, "rainfall": 48.44},
        {"name": "Muskmelon", "nitrogen": 100.32, "phosphorus": 17.72, "potassium": 50.08, "temperature": 28.66, "humidity": 92.34, "pH": 6.36, "rainfall": 24.69},
        {"name": "Orange", "nitrogen": 19.58, "phosphorus": 16.55, "potassium": 10.01, "temperature": 22.77, "humidity": 92.50, "pH": 7.01, "rainfall": 110.41},
        {"name": "Papaya", "nitrogen": 49.88, "phosphorus": 59.05, "potassium": 50.04, "temperature": 33.72, "humidity": 92.40, "pH": 6.74, "rainfall": 142.63},
        {"name": "Pigeonpeas", "nitrogen": 20.73, "phosphorus": 67.73, "potassium": 20.29, "temperature": 27.74, "humidity": 48.06, "pH": 5.79, "rainfall": 149.46},
        {"name": "Pomegranate", "nitrogen": 18.87, "phosphorus": 18.75, "potassium": 40.21, "temperature": 21.84, "humidity": 90.13, "pH": 6.43, "rainfall": 107.53},
        {"name": "Rice", "nitrogen": 79.89, "phosphorus": 47.58, "potassium": 39.87, "temperature": 23.69, "humidity": 82.27, "pH": 6.43, "rainfall": 236.18},
        {"name": "Watermelon", "nitrogen": 99.42, "phosphorus": 17.00, "potassium": 50.22, "temperature": 25.59, "humidity": 85.16, "pH": 6.50, "rainfall": 50.79},
        {"name": "Kidneybeans", "nitrogen": 20.75, "phosphorus": 67.54, "potassium": 20.05, "temperature": 20.05, "humidity": 21.61, "pH": 5.78, "rainfall": 105.92}
    ];

    # Display Team Cards
    cols = st.columns(3)  
    for index, member in enumerate(cropData):
        with cols[index % 3]:
            st.markdown(f"**    **")
            st.markdown(f"**{member['name']}**")
            st.markdown(f"Nitrogen: {member['nitrogen']}")
            st.markdown(f"Phosphorus: {member['phosphorus']}")
            st.markdown(f"Potassium: {member['potassium']}")
            st.markdown(f"Temperature: {member['temperature']}")
            st.markdown(f"pH: {member['pH']}")
            st.markdown(f"Rainfall: {member['rainfall']}")

# # FARMING GUIDE Page 
# elif app_mode == "FARMING GUIDE":
#     st.markdown("""
#          <h1 style='text-align: center; color: green;'>🌿 CROP FARMING GUIDE 🌿</h1>
#     """, unsafe_allow_html=True)

# cropGuide = [
#         {"name": "Maize Cultivation Guide", 
#            "Introduction": "Maize (Zea mays), also known as corn, is a key cereal crop widely cultivated for its grains. This guide covers the complete process for cultivating maize from seed selection to harvesting.",
#         "Materials Required": "- High-quality maize seeds (hybrid or improved varieties)\n- Fertilizers (Nitrogen, Phosphorus, Potassium)\n- Machinery (tractors, hand tools, seed planters)\n- Pest control (herbicides, insecticides)\n- Irrigation equipment (drip or furrow irrigation)",
#         "Soil Preparation": "Maize thrives in well-drained loam soils with a pH of 5.8 to 7.0. Till the soil to improve aeration and break up clods.",
#         "Seed Selection & Treatment": "Choose high-yielding, drought-resistant varieties. Treat seeds with fungicides or insecticides for protection.",
#         "Field Preparation": "Level the field for even water distribution. Optimize row spacing for maximum sunlight exposure.",
#         "Planting Time": "Typically planted at the beginning of the rainy season, between April and June, depending on the region.",
#         "Spacing & Depth": "Plant seeds at 20-25 cm within rows and 60-75 cm between rows, at a depth of 2-5 cm.",
#         "Seeding Methods": "- **Direct Seeding:** Plant seeds manually or with seed planters.",
#         "Watering Requirements": "Requires regular watering, especially during silking and tasseling. Use irrigation if rain is insufficient.",
#         "Nutrient Management": "Apply fertilizers in split doses: at planting, early growth, and tasseling stages.",
#         "Weed Control": "Manual weeding, hoeing, or herbicides. First weeding at 15-20 days after planting, followed by another at 30-40 days.",
#         "Pest & Disease Management": "Monitor for maize borers, armyworms, and aphids. Use pesticides and integrated pest management (IPM).",
#         "Harvesting": "Harvest when maize ears mature and husks dry. Moisture content should be 20-25%. Use handpicking or mechanical harvesters.",
#         "Post-Harvest Management": "Dry grains to 13-14% moisture. Shell, clean, and store properly.",
#         "Storage Conditions": "Store in a cool, dry place with ventilation to prevent mold and pests.",
#         "Processing": "If needed, dry and mill the maize for further use.",
#         "Challenges & Solutions": "Common issues: weather variability, pests, and water scarcity. Solutions: IPM, soil moisture monitoring, and resilient varieties."
#         },
        
#         {"name": "Rice Cultivation Guide", 
#             "Introduction": "Rice Oryza sativa is a staple food crop in many parts of the world. This guide covers the complete process of cultivating rice from seed selection to harvesting.",
#             "Materials Required": "- High-quality seeds\n- Fertilizers (Nitrogen, Phosphorus, Potassium)\n- Irrigation system\n- Machinery (tractors, transplanting machines, sickles)\n- Pest control (herbicides, pesticides)", 
#             "Soil Preparation": "Rice grows best in clay or clay-loam soils with pH 5.5 to 6.5. Till the soil and level the field for even water distribution.", 
#             "Seed Selection & Treatment": "Use high-yielding, pest-resistant seeds. Treat them with fungicides or insecticides to prevent infestations.", 
#             "Field Preparation": "Level the field and create bunds (raised edges) to retain water.", 
#             "Planting Time": "Plant at the onset of the rainy season, usually from May to June depending on the region.", 
#             "Spacing & Depth": "For transplanting, use 20x15 cm spacing. For direct seeding, plant 2-3 cm deep.",
#             "Seeding Methods": "- **Direct Seeding:** Broadcasting seeds or planting in rows.\n- **Transplanting:** Grow in a nursery and transfer seedlings after 20-30 days.",
#             "Watering Requirements": "Maintain 5-10 cm of water during growth. Reduce water at the grain ripening stage.",
#             "Nutrient Management": "Apply fertilizers in split doses: at planting, during tillering, and at panicle initiation.",
#             "Weed Control": "Use manual weeding or herbicides. Weed 15-20 days after transplanting, then again at 40 days.",
#             "Pest & Disease Management": "Watch for pests like stem borers and leafhoppers. Use pesticides and integrated pest management (IPM) practices.",
#             "Harvesting": "Harvest when grains turn golden-yellow and 80-90% of grains are mature. Use sickles for small farms or mechanical harvesters for efficiency.",
#             "Post-Harvest Management": "Dry grains to 14% moisture, thresh, winnow, and store in a cool, dry place to prevent spoilage.",
#             "Challenges & Solutions": "Common issues include adverse weather, pests, and water scarcity. Use IPM, monitor water levels, and diversify crop varieties to mitigate risks."
#         },
        
#         {"name": "Jute Cultivation Guide",
#             "Introduction": "Jute is a fibrous crop mainly grown for its strong, natural fibers, widely used in textiles and packaging. This guide covers the complete process for cultivating jute from seed selection to harvesting.",
#             "Materials Required": "- High-quality, certified jute seeds (Corchorus olitorius or Corchorus capsularis)\n- Organic compost, nitrogen, phosphorus, and potassium fertilizers\n- Hand tools or tractors for soil preparation\n- Herbicides and pesticides for pest control\n- Irrigation system for controlled watering",
#             "Soil Preparation": "Jute grows best in loamy, sandy-loam soils with good drainage and a pH range of 6.0 to 7.5. Prepare the soil by plowing and leveling it to break up clods and ensure good seedbed preparation.",
#             "Seed Selection & Treatment": "Choose high-yielding and disease-resistant seed varieties. Soak seeds in water for 24 hours before planting to encourage germination.",
#             "Field Preparation": "Clear and level the field for uniform water distribution. Create small bunds around the field if flooding is expected.",
#             "Planting Time": "Jute is usually planted with the arrival of the monsoon, typically between March and May.",
#             "Spacing & Depth": "Sow seeds in rows with a spacing of 25-30 cm between rows. Plant seeds 1-2 cm deep for optimal germination.",
#             "Seeding Methods": "- **Broadcasting:** Scatter seeds evenly over the field.\n- **Row Planting:** Sow seeds in rows, which facilitates weeding and other management activities.",
#             "Watering Requirements": "Jute requires regular moisture; maintain adequate moisture, especially during the early growth phase. Avoid waterlogging by ensuring proper drainage, particularly after heavy rains.",
#             "Nutrient Management": "Apply a basal dose of nitrogen, phosphorus, and potassium fertilizers at planting. Additional nitrogen can be applied after thinning, about 20-25 days after sowing.",
#             "Weed Control": "Perform manual weeding or apply selective herbicides as needed, especially in the early stages. Conduct the first weeding 15-20 days after sowing, followed by another after 30-40 days.",
#             "Pest & Disease Management": "Monitor for common pests like jute hairy caterpillars and aphids. Use pesticides or integrated pest management (IPM) practices to control pests and diseases like stem rot and anthracnose.",
#             "Harvesting": "Harvest jute when the plants are 10-12 feet tall and the lower leaves start to yellow, typically 100-120 days after planting. Cut the plants close to the base using a sickle or knife. For best fiber quality, harvest before the plants begin to flower.",
#             "Post-Harvest Management": "Bundle the harvested jute plants and submerge them in clean, slow-moving water for retting (fermentation process to loosen the fibers). Retting usually takes 10-15 days; check fiber separation regularly.",
#             "Challenges & Solutions": "Common issues include water availability, pest infestations, and improper retting. Use efficient irrigation and pest control methods, and monitor water levels carefully during retting to ensure fiber quality."
#         },

#         {"name": "Cotton Cultivation Guide",
#             "Introduction": "Cotton is a major fiber crop valued for its soft, fluffy fibers used in textiles. This guide covers the complete process for cultivating cotton from seed selection to harvesting.",
#             "Materials Required": "- High-quality, certified cotton seeds (e.g., Bt cotton or other pest-resistant varieties)\n- Nitrogen, phosphorus, potassium, and micronutrient fertilizers\n- Drip or furrow irrigation system\n- Herbicides and pesticides for pest control\n- Plows, tractors, and sprayers for field preparation and maintenance",
#             "Soil Preparation": "Cotton grows best in well-drained sandy-loam soils with a pH of 6.0 to 7.5. Prepare the field by deep plowing, followed by harrowing to break clods and smooth the surface.",
#             "Seed Selection & Treatment": "Choose high-yielding, pest-resistant seed varieties. Treat seeds with fungicides or insecticides to protect against soil-borne diseases and early pest infestations.",
#             "Field Preparation": "Create furrows or beds for planting, depending on irrigation method. Ensure good drainage to prevent waterlogging, which cotton is sensitive to.",
#             "Planting Time": "Cotton is typically planted in spring, from March to May, depending on the region and temperature.",
#             "Spacing & Depth": "Plant seeds 3-5 cm deep, with a spacing of 75-100 cm between rows and 25-30 cm between plants.",
#             "Seeding Methods": "- **Direct Seeding:** Plant seeds directly into prepared furrows or beds using seed drills or by hand.",
#             "Watering Requirements": "Cotton requires consistent moisture, especially during the flowering and boll formation stages. Use drip or furrow irrigation to maintain adequate soil moisture, particularly during dry spells.",
#             "Nutrient Management": "Apply basal fertilizer with phosphorus and potassium at planting. Apply nitrogen in split doses: one-third at planting, one-third during vegetative growth, and one-third at flowering.",
#             "Weed Control": "Use manual weeding, hoeing, or herbicides to control weeds, particularly during early growth stages. Perform weeding about 20-30 days after planting and again if necessary at 45 days.",
#             "Pest & Disease Management": "Monitor for common pests like bollworms, aphids, and whiteflies. Use integrated pest management (IPM) practices, including biological controls, to minimize pesticide use.",
#             "Harvesting": "Harvest cotton when the bolls are fully open and the fibers are fluffy, typically 150-180 days after planting. Manual harvesting involves picking mature bolls by hand, while large farms use cotton-picking machines.",
#             "Post-Harvest Management": "Allow harvested cotton to dry in a shaded, ventilated area. Clean and gin the cotton to separate seeds from fiber. Store cotton fibers in a dry, well-ventilated place to avoid moisture-related damage.",
#             "Challenges & Solutions": "Common issues include pest infestations, water availability, and soil nutrient depletion. Use drought-resistant varieties, implement efficient irrigation, and follow IPM practices to manage pests."
#         },

#         {"name": "Coconut Cultivation Guide",
#             "Introduction": "The coconut palm (Cocos nucifera) is cultivated for its fruit, providing oil, milk, and fiber. This guide covers key steps from seed selection to harvesting.",
#             "Materials Required": "- High-quality coconut seedlings (dwarf or tall varieties)\n- Organic manure, NPK fertilizers\n- Drip or basin irrigation\n- Pesticides or biocontrol agents\n- Hand tools or mechanical equipment",
#             "Soil Preparation": "Coconuts thrive in well-drained sandy loam with pH 5.5-7.5. Dig 1 x 1 x 1 m pits, fill with soil, compost, and organic manure for strong root growth.",
#             "Seed Selection & Treatment": "Use disease-resistant, high-yielding seedlings. Dwarf varieties allow easy harvesting, while tall varieties are drought-resistant.",
#             "Field Preparation": "Clear weeds and debris, ensure proper drainage, and space pits as per variety needs.",
#             "Planting Time": "Best planted at the rainy season’s onset to reduce irrigation needs; can be planted year-round with irrigation.",
#             "Spacing & Depth": "Tall varieties: 7.5-9m apart; Dwarf: 6.5-7m. Ensure roots are well covered.",
#             "Seeding Methods": "Place seedlings in pits with the collar just above ground level.",
#             "Watering Requirements": "Water regularly for the first three years. Mature trees are drought-resistant but benefit from consistent irrigation.",
#             "Nutrient Management": "Apply balanced fertilizers three times a year with micronutrients like magnesium and boron. Add organic manure annually.",
#             "Weed Control": "Weed regularly, especially in early growth. Mulching helps retain moisture and suppress weeds.",
#             "Pest & Disease Management": "Control pests like rhinoceros beetles and red palm weevils using pesticides or biocontrols. Manage root wilt and bud rot with fungicides and pruning.",
#             "Harvesting": "Mature coconuts (12 months after flowering) turn brown. Harvest every 45-60 days using climbing tools or mechanical lifters.",
#             "Post-Harvest Management": "Store in a dry, ventilated area. Process copra by sun-drying or mechanical drying. Pack dried coconuts securely for transport.",
#             "Challenges & Solutions": "Drought, pests, and soil depletion can be managed with drip irrigation, pest management, and organic soil amendments."
#         },

#         {"name": "Chickpea Cultivation Guide",
#             "Introduction": "Chickpea (Cicer arietinum) is a popular legume grown for its protein-rich seeds, widely used in food production. This guide covers the complete process for cultivating chickpeas from seed selection to harvesting.",
#             "Materials Required": "- High-quality, disease-resistant chickpea seeds (desi or kabuli types)\n- Phosphorus-based fertilizers; minimal nitrogen\n- Drip or sprinkler irrigation\n- Herbicides and pesticides\n- Plows, tractors, and sprayers",
#             "Soil Preparation": "Chickpeas grow best in well-drained, loamy soils with a pH of 6.0-7.5. Plow and harrow the field for good root penetration.",
#             "Seed Selection & Treatment": "Choose high-yielding, disease-resistant seeds. Treat with rhizobium bacteria for nitrogen fixation and fungicides to prevent diseases.",
#             "Field Preparation": "Clear weeds and level the field. Space rows to allow air circulation and reduce disease risk.",
#             "Planting Time": "Best planted in cool, dry seasons, typically October-November.",
#             "Spacing & Depth": "Space plants 30-40 cm apart in rows 45-60 cm apart. Sow seeds 5-8 cm deep based on soil moisture.",
#             "Seeding Methods": "Direct seeding using seed drills or manual planting.",
#             "Watering Requirements": "Chickpeas require minimal watering but benefit from irrigation during flowering and pod filling. Avoid waterlogging.",
#             "Nutrient Management": "Apply phosphorus at planting. Use potassium and micronutrients as needed based on soil tests.",
#             "Weed Control": "Weed early and regularly, either manually or with herbicides. First weeding at 20-30 days, second at 45-50 days if needed.",
#             "Pest & Disease Management": "Monitor for pests like pod borers and aphids. Use integrated pest management (IPM) and biopesticides as needed.",
#             "Special Care During Growth": "- Seedling stage: Protect from pests, maintain moderate moisture.\n- Vegetative stage: Maintain phosphorus levels.\n- Flowering & pod-filling: Ensure adequate moisture for optimal yield.",
#             "Harvesting": "Chickpeas mature in 3-4 months. Harvest when plants yellow and pods dry. Cut by hand for small farms; use combine harvesters for large-scale farming.",
#             "Post-Harvest Management": "Sun-dry seeds to reduce moisture, thresh, and clean before storage or sale.",
#             "Storage Conditions": "Store in dry, cool places with ventilation to prevent insect infestations and spoilage.",
#             "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
#             "Challenges & Solutions": "Common issues include pests, diseases, water stress, and nutrient deficiencies. Use IPM, resistant varieties, and soil testing to mitigate risks."
#         },

#         {"name": "Pigeon Pea Cultivation Guide",
#             "Introduction": "Pigeon peas (Cajanus cajan) are a drought-resistant legume valued for their high protein content and use in various dishes. This guide covers the complete process for cultivating pigeon peas from seed selection to harvesting.",
#             "Materials Required": "- High-quality, disease-resistant pigeon pea seeds (early, medium, or late-maturing varieties)\n- Nitrogen, phosphorus, and potassium fertilizers; minimal nitrogen needed\n- Drip or furrow irrigation equipment\n- Herbicides and pesticides specific to pigeon pea pests\n- Hand tools or tractors for soil preparation, planting, and weeding",
#             "Soil Preparation": "Pigeon peas grow best in well-drained sandy loam to clay loam soils with a pH of 6.0-7.5. Plow and harrow the field to create a fine seedbed.",
#             "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suitable for your region. Treat seeds with fungicides to prevent seed-borne diseases.",
#             "Field Preparation": "Clear the field of weeds and debris, ensuring good drainage.",
#             "Planting Time": "Typically planted at the beginning of the rainy season or during the dry season in subtropical regions.",
#             "Spacing & Depth": "Space plants 30-40 cm apart in rows 60-75 cm apart. Sow seeds 3-5 cm deep, depending on soil moisture and texture.",
#             "Seeding Methods": "Direct seeding using seed drills or manual planting.",
#             "Watering Requirements": "Pigeon peas are drought-resistant but require adequate moisture during flowering and pod development. Irrigation may be necessary, especially in the first 60 days.",
#             "Nutrient Management": "Apply phosphorus and potassium at planting and top-dress with nitrogen if necessary. Organic amendments can improve soil fertility.",
#             "Weed Control": "Control weeds during early growth stages using manual weeding or herbicides. Mulching can help suppress weeds and retain soil moisture.",
#             "Pest & Disease Management": "Monitor for pests such as pod borers, aphids, and whiteflies. Implement integrated pest management (IPM) strategies, including biological controls and chemical pesticides as needed.",
#             "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain soil moisture.\n- Vegetative stage: Ensure adequate nutrients for strong growth.\n- Flowering & pod-filling: Maintain consistent moisture to maximize yield and seed quality.",
#             "Harvesting": "Pigeon peas mature in 4-6 months. Harvest when pods are mature and dry. Cut by hand for small farms or use combine harvesters for large-scale farming.",
#             "Post-Harvest Management": "Allow harvested plants to sun-dry before threshing to reduce seed moisture content.",
#             "Storage Conditions": "Store pigeon peas in a dry, cool, and well-ventilated area to prevent spoilage and insect infestations.",
#             "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags or containers.",
#             "Challenges & Solutions": "Common issues include pest infestations, diseases, water stress, and nutrient deficiencies. Use disease-resistant varieties, practice crop rotation, and apply IPM strategies to manage risks."
#         },

#         {"name": "Moth Bean Cultivation Guide",
#             "Introduction": "Moth beans (Vigna aconitifolia) are a drought-resistant legume commonly grown in arid regions. They are valued for their high protein content and culinary applications. This guide covers the complete process for cultivating moth beans from seed selection to harvesting.",
#             "Materials Required": "- High-quality, disease-resistant moth bean seeds\n- Phosphorus and potassium fertilizers; minimal nitrogen\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
#             "Soil Preparation": "Moth beans thrive in well-drained sandy loam or clay soils with a pH of 6.0-8.0. Prepare the field by plowing and harrowing for a fine seedbed.",
#             "Seed Selection & Treatment": "Choose high-yielding, drought-tolerant varieties. Treat seeds with fungicides or insecticides to prevent seed-borne diseases.",
#             "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
#             "Planting Time": "Typically planted at the onset of the monsoon season, between June and July.",
#             "Spacing & Depth": "Space plants 30-45 cm apart in rows 60-75 cm apart. Sow seeds 3-5 cm deep based on soil moisture.",
#             "Seeding Methods": "Direct seeding using seed drills or manual planting.",
#             "Watering Requirements": "Moth beans are drought-resistant but benefit from consistent moisture during flowering and pod development. Water if rainfall is insufficient.",
#             "Nutrient Management": "Apply phosphorus and potassium at planting. Use nitrogen only if soil tests indicate a deficiency. Organic amendments improve soil fertility.",
#             "Weed Control": "Control weeds early with manual weeding or herbicides. Mulching helps suppress weeds and retain soil moisture.",
#             "Pest & Disease Management": "Monitor for pests like aphids, pod borers, and leafhoppers. Use integrated pest management (IPM) strategies as needed.",
#             "Special Care During Growth": "- Seedling stage: Maintain moderate moisture and protect from pests.\n- Vegetative stage: Ensure adequate nutrients.\n- Flowering & pod-filling: Maintain moisture for optimal yield.",
#             "Harvesting": "Harvest when pods mature and dry, typically 90-120 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
#             "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
#             "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
#             "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
#             "Challenges & Solutions": "Common issues include pests, diseases, and adverse weather. Use drought-resistant varieties, IPM practices, and proper soil management to mitigate risks."
#         },

#         {"name": "Mung Bean Cultivation Guide",
#             "Introduction": "Mung beans (Vigna radiata) are small, green legumes highly valued for their nutritional content and culinary versatility. This guide covers the complete process for cultivating mung beans from seed selection to harvesting.",
#             "Materials Required": "- High-quality, disease-resistant mung bean seeds\n- Nitrogen, phosphorus, and potassium fertilizers (minimal nitrogen needed)\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
#             "Soil Preparation": "Mung beans prefer well-drained sandy loam to loamy soils with a pH of 6.0-7.5. Prepare the field by plowing and harrowing to achieve a fine seedbed.",
#             "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suitable for your climate. Treat seeds with fungicides to protect against soil-borne diseases.",
#             "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
#             "Planting Time": "Typically planted at the beginning of the rainy season or in warm, dry conditions between April and June.",
#             "Spacing & Depth": "Space plants 30-40 cm apart in rows 45-60 cm apart. Sow seeds 2-4 cm deep based on soil moisture.",
#             "Seeding Methods": "Direct seeding using seed drills or manual planting.",
#             "Watering Requirements": "Mung beans require adequate moisture, particularly during germination and flowering. Water if rainfall is insufficient, ensuring not to overwater to prevent root rot.",
#             "Nutrient Management": "Apply phosphorus and potassium at planting. Additional nitrogen may be applied if needed, but usually, the natural fixation suffices. Incorporate organic matter to improve soil fertility.",
#             "Weed Control": "Control weeds early through manual weeding or herbicides. Mulching helps suppress weeds and conserve soil moisture.",
#             "Pest & Disease Management": "Monitor for pests like aphids, beetles, and thrips. Use integrated pest management (IPM) strategies as needed.",
#             "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain adequate moisture.\n- Vegetative stage: Ensure sufficient nutrients for strong growth.\n- Flowering & pod-filling: Maintain moisture for optimal yield and quality.",
#             "Harvesting": "Harvest when pods mature and dry, typically 60-90 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
#             "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
#             "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
#             "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
#             "Challenges & Solutions": "Common issues include pests, diseases, and adverse weather. Use disease-resistant varieties, IPM practices, and proper soil and water management to mitigate risks."
#         },

#         {"name": "Black Gram Cultivation Guide",
#             "Introduction": "Black gram (Vigna mungo) is a highly nutritious legume valued for its high protein content and is widely used in various culinary dishes. This guide covers the complete process for cultivating black gram from seed selection to harvesting.",
#             "Materials Required": "- High-quality, disease-resistant black gram seeds\n- Phosphorus and potassium fertilizers (minimal nitrogen needed)\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
#             "Soil Preparation": "Black gram prefers well-drained sandy loam to clay loam soils with a pH of 6.0-7.5. Prepare the field by plowing and harrowing to create a fine seedbed.",
#             "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suitable for your climate. Treat seeds with fungicides or insecticides to protect against soil-borne diseases.",
#             "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
#             "Planting Time": "Typically planted at the beginning of the monsoon season or during warm, dry conditions between June and July.",
#             "Spacing & Depth": "Space plants 30-45 cm apart in rows 60-75 cm apart. Sow seeds 3-5 cm deep based on soil moisture.",
#             "Seeding Methods": "Direct seeding using seed drills or manual planting.",
#             "Watering Requirements": "Black gram requires adequate moisture, particularly during germination and flowering. Water if rainfall is insufficient, ensuring not to overwater to prevent root rot.",
#             "Nutrient Management": "Apply phosphorus and potassium at planting. Additional nitrogen is generally not necessary due to nitrogen fixation. Incorporate organic matter to improve soil fertility.",
#             "Weed Control": "Control weeds early through manual weeding or herbicides. Mulching helps suppress weeds and conserve soil moisture.",
#             "Pest & Disease Management": "Monitor for pests like aphids, pod borers, and thrips. Use integrated pest management (IPM) strategies as needed.",
#             "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain adequate moisture.\n- Vegetative stage: Ensure sufficient nutrients for strong growth.\n- Flowering & pod-filling: Maintain moisture for optimal yield and quality.",
#             "Harvesting": "Harvest when pods mature and dry, typically 60-90 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
#             "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
#             "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
#             "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
#             "Challenges & Solutions": "Common issues include pests, diseases, and adverse weather. Use disease-resistant varieties, IPM practices, and proper soil and water management to mitigate risks."
#         },

#         {"name": "Lentil Cultivation Guide",
#             "Introduction": "Lentils (Lens culinaris) are nutritious legumes known for their high protein and fiber content. They are widely cultivated for food and are a staple in many cuisines. This guide covers the complete process for cultivating lentils from seed selection to harvesting.",
#             "Materials Required": "- High-quality, disease-resistant lentil seeds\n- Phosphorus and potassium fertilizers (minimal nitrogen needed)\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
#             "Soil Preparation": "Lentils prefer well-drained loamy or sandy soils with a pH of 6.0-7.5. Prepare the field by plowing and harrowing to create a fine seedbed.",
#             "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suited to your region. Treat seeds with fungicides or insecticides to protect against seed-borne diseases.",
#             "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
#             "Planting Time": "Lentils are typically planted in early spring or late winter, depending on the climate, when soil temperatures reach around 10-15°C (50-59°F).",
#             "Spacing & Depth": "Space plants 25-30 cm apart in rows 45-60 cm apart. Sow seeds 2-3 cm deep based on soil moisture.",
#             "Seeding Methods": "Direct seeding using seed drills or manual planting.",
#             "Watering Requirements": "Lentils are drought-tolerant but need adequate moisture during germination and pod development. Water if rainfall is insufficient, particularly during flowering and seed filling.",
#             "Nutrient Management": "Apply phosphorus and potassium at planting. Additional nitrogen is typically not needed due to nitrogen fixation. Incorporate organic matter to enhance soil fertility.",
#             "Weed Control": "Control weeds during early growth using manual weeding or herbicides. Mulching can also help suppress weeds and retain soil moisture.",
#             "Pest & Disease Management": "Monitor for pests such as aphids, lygus bugs, and root rots. Implement integrated pest management (IPM) strategies as needed.",
#             "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain adequate moisture.\n- Vegetative stage: Ensure sufficient nutrients for strong growth.\n- Flowering & pod-filling: Maintain moisture for optimal yield and quality.",
#             "Harvesting": "Harvest when pods turn brown and dry, typically 80-100 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
#             "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
#             "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
#             "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
#             "Challenges & Solutions": "Common issues include pests, diseases, and variable weather. Use disease-resistant varieties, IPM practices, and proper soil and water management to mitigate risks."
#         },

#         {"name": "Pomegranate Cultivation Guide",
#             "Introduction": "Pomegranates (Punica granatum) are nutritious fruits known for their health benefits and vibrant flavor. They are cultivated in many parts of the world and thrive in warm climates. This guide covers the complete process for cultivating pomegranates from planting to harvesting.",
#             "Materials Required": "- High-quality pomegranate seeds or healthy seedlings from reputable nurseries\n- Balanced fertilizers with nitrogen, phosphorus, and potassium\n- Drip irrigation systems or furrow irrigation\n- Insecticides and fungicides for pest and disease management\n- Hand tools or tractors for planting, pruning, and maintenance",
#             "Soil Preparation": "Pomegranates prefer well-drained, sandy loam to loamy soils with a pH of 5.5 to 7.0. Prepare the planting site by plowing and incorporating organic matter.",
#             "Seed Selection & Treatment": "Choose disease-resistant varieties suitable for your region's climate. If using seeds, soak them overnight in water before planting to improve germination rates.",
#             "Field Preparation": "Clear the site of weeds, rocks, and debris to ensure a clean planting environment.",
#             "Planting Time": "Pomegranates are typically planted in spring after the last frost.",
#             "Spacing & Depth": "Space plants 5-8 feet apart to allow for proper growth and air circulation. Plant seeds or seedlings at a depth of 1-2 inches, ensuring good soil contact.",
#             "Seeding Methods": "Direct Seeding: Sow seeds directly into the prepared site. Transplanting: For seedlings, dig a hole slightly larger than the root ball and backfill with soil.",
#             "Watering Requirements": "Pomegranates require regular watering, especially during the establishment phase; once established, they are drought-tolerant. Water deeply but infrequently to encourage deep root growth.",
#             "Nutrient Management": "Apply a balanced fertilizer during the growing season, typically in early spring and again in late summer. Incorporate organic compost to improve soil fertility.",
#             "Weed Control": "Control weeds using mulching and manual weeding to reduce competition for nutrients.",
#             "Pest & Disease Management": "Monitor for pests such as aphids, whiteflies, and pomegranate butterflies. Implement integrated pest management (IPM) strategies, including the use of natural predators and organic pesticides.",
#             "Special Care During Growth": "- Seedling stage: Protect young plants from extreme weather and pests. Use mulch to retain moisture.\n- Vegetative stage: Regularly check for nutrient deficiencies and pest infestations; apply fertilizers as needed.\n- Flowering & fruit development: Ensure adequate water during flowering and fruit set to promote healthy development.",
#             "Harvesting": "Pomegranates are typically ready for harvest 5-7 months after flowering, when the fruit has a deep color and makes a metallic sound when tapped. Use sharp pruning shears to cut the fruit from the tree, avoiding damage to the branches and other fruit.",
#             "Post-Harvest Management": "Handle fruits gently to prevent bruising; store in a cool, dry place.",
#             "Storage Conditions": "Store pomegranates in a cool, dry environment; they can last several weeks to months in proper conditions.",
#             "Processing & Packaging": "Clean and sort harvested fruits, discarding any damaged or rotten ones. Pack fruits in breathable containers to maintain quality during storage.",
#             "Challenges & Solutions": "Common issues include susceptibility to pests, diseases, and environmental stresses such as drought or excessive moisture. Use disease-resistant varieties, implement proper irrigation practices, and monitor pest populations to mitigate challenges."
#         },

#         {"name": "Kidney Bean Cultivation Guide",
#             "Introduction": "Kidney beans (Phaseolus vulgaris) are a high-protein legume commonly used in various cuisines. This guide covers the complete process for cultivating kidney beans from seed selection to harvesting.",
#             "Materials Required": "- High-quality, disease-resistant kidney bean seeds\n- Phosphorus and potassium fertilizers; minimal nitrogen as beans fix their own nitrogen\n- Drip or sprinkler irrigation\n- Herbicides and pesticides for common kidney bean pests\n- Hand tools or tractors for soil preparation, planting, and weeding",
#             "Soil Preparation": "Kidney beans thrive in well-drained, loamy soils with a pH between 6.0 and 7.0. Prepare the field by plowing and harrowing to create a fine tilth for easy root penetration.",
#             "Seed Selection & Treatment": "Choose high-yielding, disease-resistant seed varieties. Treat seeds with fungicides or insecticides to protect against early soil-borne diseases and pests.",
#             "Field Preparation": "Clear the field of weeds and debris, then level it. Mark rows with adequate spacing for air circulation and sunlight penetration.",
#             "Planting Time": "Kidney beans are typically planted in spring when soil temperatures reach 15°C (59°F) and there is no risk of frost.",
#             "Spacing & Depth": "Plant seeds 3-5 cm deep, with 8-10 cm between plants and 45-60 cm between rows.",
#             "Seeding Methods": "Direct Seeding: Sow seeds directly into the field by hand or using a seed drill.",
#             "Watering Requirements": "Kidney beans need regular watering, particularly during flowering and pod development. Avoid overwatering, as beans are sensitive to waterlogging.",
#             "Nutrient Management": "Apply phosphorus and potassium at planting. Limit nitrogen since kidney beans fix atmospheric nitrogen. Supplement micronutrients if soil tests indicate deficiencies.",
#             "Weed Control": "Weed control is essential, particularly in the early stages. Use manual weeding or herbicides as needed. Mulching around plants can help retain moisture and suppress weeds.",
#             "Pest & Disease Management": "Monitor for pests like aphids, leafhoppers, and bean beetles. Use integrated pest management (IPM) practices and apply pesticides if necessary. Prevent diseases like root rot and blight by practicing crop rotation and avoiding waterlogged soil.",
#             "Special Care During Growth": "- Seedling stage: Ensure moderate soil moisture and protect seedlings from pests.\n- Vegetative stage: Maintain nutrient levels to support robust leaf and stem growth.\n- Flowering & pod-filling stage: Provide consistent moisture during pod development to enhance yield and seed quality.",
#             "Harvesting": "Harvest kidney beans when the pods are fully mature and dry, usually 90-120 days after planting. For small farms, harvest by hand by pulling up the entire plant. For larger farms, use a combine harvester to gather beans efficiently.",
#             "Post-Harvest Management": "Allow the harvested plants to dry in the sun to reduce moisture in the seeds. Thresh the beans to separate them from the pods, then clean the seeds.",
#             "Storage Conditions": "Store kidney beans in a dry, well-ventilated place to prevent mold and insect infestations.",
#             "Processing & Packaging": "Clean and grade the beans for quality assurance before packaging. Pack beans in breathable bags or containers to maintain quality during storage.",
#             "Challenges & Solutions": "Common issues include susceptibility to pests, diseases, and nutrient imbalances. Use disease-resistant seeds, monitor soil health, and apply IPM practices to control pests and diseases effectively."
#         },

#         {"name": "Banana Cultivation Guide",
#             "Introduction": "Bananas (Musa spp.) are tropical fruits renowned for their sweet flavor and nutritional benefits. They thrive in warm, humid climates and are cultivated worldwide for both commercial and home production. This guide outlines the complete process for cultivating bananas, from planting to harvesting.",
#             "Materials Required": "- Healthy banana suckers or tissue-cultured plantlets\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic matter such as compost\n- Drip or sprinkler irrigation systems for adequate moisture management\n- Insecticides and fungicides to manage pests and diseases\n- Hand tools (shovels, pruners) or tractors for planting, maintenance, and harvesting",
#             "Soil Preparation": "Bananas prefer well-drained, rich loamy soils with a pH of 5.5 to 7.0. Prepare the soil by plowing and incorporating organic matter to improve fertility and drainage.",
#             "Plant Selection & Treatment": "Select disease-free suckers from healthy parent plants or obtain tissue-cultured plantlets from a reputable source. If using suckers, cut them from the parent plant with a clean knife to avoid contamination.",
#             "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
#             "Planting Time": "The ideal time to plant bananas is at the beginning of the rainy season or during the warmer months.",
#             "Spacing & Depth": "Space plants 8-10 feet apart in rows that are 10-12 feet apart to allow for proper growth and air circulation. Plant suckers or plantlets at the same depth they were growing in the nursery.",
#             "Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots and backfill gently to avoid air pockets.",
#             "Watering Requirements": "Bananas require consistent moisture; irrigate regularly, especially during dry spells. Aim for 1-2 inches of water per week.",
#             "Nutrient Management": "Apply a balanced fertilizer in early spring and again mid-season. Add compost or organic mulch to enhance soil fertility.",
#             "Weed Control": "Control weeds using mulching, which also helps retain soil moisture, and manual weeding to reduce competition for nutrients.",
#             "Pest & Disease Management": "Monitor for pests such as banana weevils and aphids. Manage diseases like Panama disease and leaf spot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of biological pest control methods.",
#             "Special Care During Growth": "- Seedling stage: Protect young plants from extreme weather and pests; consider using shade cloth if necessary.\n- Vegetative stage: Regularly check for nutrient deficiencies, especially potassium and magnesium, and address them promptly.\n- Flowering & fruit development stage: Ensure adequate water supply during flowering and fruit development to support healthy fruit formation.",
#             "Harvesting": "Bananas are typically ready for harvest 9-12 months after planting, depending on the variety and growing conditions. Harvest when the fruit is plump, green, and the angle between the fruit and the stalk becomes more pronounced. Use a sharp knife or machete to cut the entire bunch from the plant. Handle the fruit carefully to avoid bruising.",
#             "Post-Harvest Management": "Remove any excess leaves and handle harvested bananas gently to prevent damage. Store them in a cool, shaded area.",
#             "Storage Conditions": "Store bananas at room temperature until they ripen. Avoid exposure to direct sunlight or excessive heat.",
#             "Processing & Packaging": "If needed, bananas can be processed into products like banana chips or puree. Pack bananas in breathable boxes to allow for airflow and reduce spoilage during transport.",
#             "Challenges & Solutions": "Common issues include susceptibility to pests and diseases, environmental stresses, and improper watering. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
#         },

#         {"name": "Banana Cultivation Guide",
#             "Introduction": "Bananas (Musa spp.) are tropical fruits renowned for their sweet flavor and nutritional benefits. They thrive in warm, humid climates and are cultivated worldwide for both commercial and home production. This guide outlines the complete process for cultivating bananas, from planting to harvesting.",
#             "Materials Required": "- Healthy banana suckers or tissue-cultured plantlets\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic matter such as compost\n- Drip or sprinkler irrigation systems for adequate moisture management\n- Insecticides and fungicides to manage pests and diseases\n- Hand tools (shovels, pruners) or tractors for planting, maintenance, and harvesting",
#             "Soil Preparation": "Bananas prefer well-drained, rich loamy soils with a pH of 5.5 to 7.0. Prepare the soil by plowing and incorporating organic matter to improve fertility and drainage.",
#             "Plant Selection & Treatment": "Select disease-free suckers from healthy parent plants or obtain tissue-cultured plantlets from a reputable source. If using suckers, cut them from the parent plant with a clean knife to avoid contamination.",
#             "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
#             "Planting Time": "The ideal time to plant bananas is at the beginning of the rainy season or during the warmer months.",
#             "Spacing & Depth": "Space plants 8-10 feet apart in rows that are 10-12 feet apart to allow for proper growth and air circulation. Plant suckers or plantlets at the same depth they were growing in the nursery.",
#             "Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots and backfill gently to avoid air pockets.",
#             "Watering Requirements": "Bananas require consistent moisture; irrigate regularly, especially during dry spells. Aim for 1-2 inches of water per week.",
#             "Nutrient Management": "Apply a balanced fertilizer in early spring and again mid-season. Add compost or organic mulch to enhance soil fertility.",
#             "Weed Control": "Control weeds using mulching, which also helps retain soil moisture, and manual weeding to reduce competition for nutrients.",
#             "Pest & Disease Management": "Monitor for pests such as banana weevils and aphids. Manage diseases like Panama disease and leaf spot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of biological pest control methods.",
#             "Special Care During Growth": "- Seedling stage: Protect young plants from extreme weather and pests; consider using shade cloth if necessary.\n- Vegetative stage: Regularly check for nutrient deficiencies, especially potassium and magnesium, and address them promptly.\n- Flowering & fruit development stage: Ensure adequate water supply during flowering and fruit development to support healthy fruit formation.",
#             "Harvesting": "Bananas are typically ready for harvest 9-12 months after planting, depending on the variety and growing conditions. Harvest when the fruit is plump, green, and the angle between the fruit and the stalk becomes more pronounced. Use a sharp knife or machete to cut the entire bunch from the plant. Handle the fruit carefully to avoid bruising.",
#             "Post-Harvest Management": "Remove any excess leaves and handle harvested bananas gently to prevent damage. Store them in a cool, shaded area.",
#             "Storage Conditions": "Store bananas at room temperature until they ripen. Avoid exposure to direct sunlight or excessive heat.",
#             "Processing & Packaging": "If needed, bananas can be processed into products like banana chips or puree. Pack bananas in breathable boxes to allow for airflow and reduce spoilage during transport.",
#             "Challenges & Solutions": "Common issues include susceptibility to pests and diseases, environmental stresses, and improper watering. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
#         },


#         {"name": "Grape Cultivation Guide",
#             "Introduction": "Grapes (Vitis vinifera and other species) are versatile fruits used for fresh eating, drying (raisins), and wine production. They thrive in temperate climates and require specific growing conditions to produce high-quality fruit. This guide outlines the complete process for cultivating grapes, from planting to harvesting.",
#             "Materials Required": "- Quality grapevines, either bare-root or potted, from reputable nurseries\n- Balanced fertilizers containing nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems for efficient moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (pruners, shovels) or tractors for planting, maintenance, and harvesting",
#             "Soil Preparation": "Grapes prefer well-drained, sandy loam or clay loam soils with a pH of 6.0 to 6.8. Prepare the soil by tilling and incorporating organic matter to enhance fertility and drainage.",
#             "Plant Selection & Treatment": "Select disease-resistant grape varieties suitable for your climate and purpose (table grapes, wine grapes, etc.). Inspect vines for signs of disease or damage before planting.",
#             "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
#             "Planting Time": "The ideal time to plant grapes is in early spring after the last frost or in the fall before the ground freezes.",
#             "Spacing & Depth": "Space vines 6-10 feet apart in rows that are 8-10 feet apart to allow for proper air circulation and growth. Plant vines at the same depth they were growing in the nursery.",
#             "Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, backfill gently, and water thoroughly after planting.",
#             "Watering Requirements": "Grapes require regular watering during the first year to establish roots. Once established, they are drought-tolerant but still benefit from supplemental irrigation during dry spells, especially during fruit development.",
#             "Nutrient Management": "Apply a balanced fertilizer in early spring and again mid-season. Use organic compost to improve soil health.",
#             "Weed Control": "Control weeds through mulching, hand weeding, or the use of herbicides to reduce competition for nutrients and moisture.",
#             "Pest & Disease Management": "Monitor for pests such as grapevine moths, aphids, and spider mites. Manage diseases like powdery mildew and downy mildew with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and natural predators.",
#             "Special Care During Growth": "- Young Vine Stage: Protect young vines from extreme weather and pests; use support stakes or trellises to help young plants grow upward.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to encourage a strong structure and air circulation.\n- Flowering & Fruit Development Stage: Ensure consistent moisture during flowering and fruit set to maximize yield and fruit quality. Thin clusters if necessary to promote larger fruit size.",
#             "Harvesting": "Grapes are typically ready for harvest 4-6 months after flowering, depending on the variety. They should be harvested when fully ripe, showing deep color and sweet flavor. Use sharp pruning shears to cut clusters from the vine. Handle the fruit carefully to avoid bruising.",
#             "Post-Harvest Management": "Remove any damaged or rotten grapes and store them in a cool, shaded area.",
#             "Storage Conditions": "Store grapes in a cool, dry place. Refrigeration can extend their shelf life, but they should be kept in breathable containers.",
#             "Processing & Packaging": "If needed, grapes can be processed into products like grape juice, jelly, or wine. Pack grapes in breathable containers to allow airflow and reduce spoilage during transport.",
#             "Challenges & Solutions": "Common issues include susceptibility to pests and diseases, climate-related issues, and improper watering. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
#         },

#         {"name": "Muskmelon Cultivation Guide",
#             "Introduction": "Muskmelons (Cucumis melo var. cantaloupe) are sweet, aromatic fruits known for their juicy flesh and distinctive netted skin. They thrive in warm climates and are popular for their refreshing taste. This guide outlines the complete process for cultivating muskmelons, from planting to harvesting.",
#             "Materials Required": "- Quality muskmelon seeds or seedlings from reputable sources\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic compost\n- Drip or overhead irrigation systems for efficient moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, hoes, pruners) or tractors for planting, maintenance, and harvesting",
#             "Soil Preparation": "Muskmelons prefer well-drained, sandy loam or loamy soils with a pH of 6.0 to 6.8. Prepare the soil by tilling and mixing in organic matter to enhance drainage and fertility.",
#             "Plant Selection & Treatment": "Choose disease-resistant varieties suited for your climate and market. If using seeds, soak them in water for a few hours before planting to improve germination rates.",
#             "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
#             "Planting Time": "The ideal time to plant muskmelons is after the last frost date when soil temperatures are consistently above 70°F (21°C).",
#             "Spacing & Depth": "Space muskmelon plants 3-4 feet apart in rows that are 6-8 feet apart to allow for sprawling vines. Plant seeds or seedlings at a depth of about 1 inch.",
#             "Seeding/Transplanting Methods": "Direct Seeding: Plant seeds directly into the ground after the soil warms up. Transplanting: Start seedlings indoors and transplant them once they are strong enough.",
#             "Watering Requirements": "Muskmelons need consistent moisture, especially during germination and fruit development. Aim for about 1-2 inches of water per week, adjusting for rainfall.",
#             "Nutrient Management": "Apply a balanced fertilizer at planting and again when vines begin to run. Use organic compost or mulch to enhance soil health.",
#             "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
#             "Pest & Disease Management": "Monitor for pests such as aphids, cucumber beetles, and spider mites. Manage diseases like powdery mildew and downy mildew with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of biological controls.",
#             "Special Care During Growth": "- Seedling Stage: Protect young plants from pests and extreme weather. Use row covers if necessary to protect against pests and frost.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Support vines if necessary, especially when fruit begins to develop.\n- Fruit Development Stage: Ensure adequate water supply during fruit development to promote healthy growth and sweetness. Avoid watering directly on the fruit to prevent rot.",
#             "Harvesting": "Muskmelons are typically ready for harvest 70-90 days after planting. Indicators include a change in color from green to yellow at the blossom end and a sweet aroma. Use a sharp knife or pruning shears to cut the fruit from the vine, leaving a short stem attached to the melon.",
#             "Post-Harvest Management": "Handle harvested muskmelons gently to avoid bruising. Store them in a cool, shaded area.",
#             "Storage Conditions": "Store muskmelons at room temperature until they are fully ripe. Once ripe, they can be refrigerated for a short period to extend freshness.",
#             "Processing & Packaging": "If needed, muskmelons can be processed into smoothies, sorbets, or fruit salads. Pack muskmelons in breathable containers to help maintain quality during storage and transport.",
#             "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses such as drought or excessive moisture, and improper watering practices. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
#         },

#         {"name": "Apple Cultivation Guide",
#             "Introduction": "Apples (Malus domestica) are one of the most popular fruits worldwide, appreciated for their taste, versatility, and nutritional value. They grow best in temperate climates and can be cultivated in various soil types. This guide outlines the complete process for cultivating apples, from planting to harvesting.",
#             "Materials Required": "- Quality apple tree seedlings or grafted varieties from reputable nurseries\n- Balanced fertilizers containing nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for effective moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
#             "Soil Preparation": "Apples prefer well-drained, loamy soils with a pH of 6.0 to 7.0. Prepare the soil by tilling and incorporating organic matter to improve fertility and drainage.",
#             "Plant Selection & Treatment": "Choose disease-resistant apple varieties suited to your climate, considering factors such as fruit flavor and harvest time. Inspect seedlings for signs of disease or damage before planting.",
#             "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
#             "Planting Time": "The best time to plant apple trees is in the fall or early spring when the trees are dormant.",
#             "Spacing & Depth": "Space dwarf varieties 4-6 feet apart and standard varieties 10-15 feet apart to allow for proper growth and air circulation. Plant trees at a depth that matches their nursery height, ensuring the graft union is above soil level.",
#             "Seeding/Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, place the tree in the hole, backfill gently, and water thoroughly after planting.",
#             "Watering Requirements": "Water young apple trees regularly to establish roots, especially during dry spells. Once established, they are drought-tolerant but benefit from deep watering during fruit development.",
#             "Nutrient Management": "Apply a balanced fertilizer in early spring and again in mid-season. Use organic compost to enhance soil health.",
#             "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
#             "Pest & Disease Management": "Monitor for pests such as codling moths, aphids, and spider mites. Manage diseases like apple scab and powdery mildew with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
#             "Special Care During Growth": "- Young Tree Stage: Protect young trees from extreme weather and pests; consider using tree guards to prevent animal damage.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to shape trees and encourage a strong structure.\n- Flowering and Fruit Development Stage: Ensure consistent moisture during flowering and fruit set to maximize yield and fruit quality. Thin fruit if necessary to promote larger apples.",
#             "Harvesting": "Apples are typically ready for harvest 4-6 months after flowering, depending on the variety. Indicators include a change in color, firm texture, and ease of detachment from the tree. Use sharp pruning shears to cut apples from the tree, leaving a short stem attached to the fruit.",
#             "Post-Harvest Management": "Handle harvested apples gently to avoid bruising. Store them in a cool, shaded area.",
#             "Storage Conditions": "Store apples in a cool, dark place. They can be refrigerated to extend their shelf life.",
#             "Processing & Packaging": "If needed, apples can be processed into applesauce, cider, or dried slices. Pack apples in breathable containers to help maintain quality during storage and transport.",
#             "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or frost), and improper pruning techniques. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
#         },

#         {"name": "Orange Cultivation Guide",
#             "Introduction": "Oranges (Citrus sinensis) are one of the most popular citrus fruits, valued for their sweet, juicy flesh and high vitamin C content. They thrive in warm, subtropical to tropical climates. This guide outlines the complete process for cultivating oranges, from planting to harvesting.",
#             "Materials Required": "- Quality orange tree seedlings or grafted varieties from reputable nurseries\n- Citrus-specific fertilizers containing nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for efficient moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
#             "Soil Preparation": "Oranges prefer well-drained, sandy loam or clay loam soils with a pH of 6.0 to 7.5. Prepare the soil by tilling and incorporating organic matter to improve fertility and drainage.",
#             "Plant Selection & Treatment": "Choose disease-resistant orange varieties suited to your climate, considering factors such as fruit flavor and harvest time. Inspect seedlings for signs of disease or damage before planting.",
#             "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
#             "Planting Time": "The best time to plant orange trees is in the spring after the danger of frost has passed.",
#             "Spacing & Depth": "Space trees 12-25 feet apart, depending on the rootstock and tree variety, to allow for proper growth and air circulation. Plant trees at a depth that matches their nursery height, ensuring the graft union is above soil level.",
#             "Seeding/Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, place the tree in the hole, backfill gently, and water thoroughly after planting.",
#             "Watering Requirements": "Water young orange trees regularly to establish roots, especially during dry spells. Mature trees require deep watering during dry periods.",
#             "Nutrient Management": "Apply a citrus-specific fertilizer in early spring and again in mid-season. Use organic compost to enhance soil health.",
#             "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
#             "Pest & Disease Management": "Monitor for pests such as aphids, spider mites, and citrus leaf miners. Manage diseases like citrus canker and root rot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
#             "Special Care During Growth": "- Young Tree Stage: Protect young trees from extreme weather and pests; consider using tree guards to prevent animal damage.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to shape trees and encourage a strong structure.\n- Flowering and Fruit Development Stage: Ensure consistent moisture during flowering and fruit set to maximize yield and fruit quality. Thin fruit if necessary to promote larger oranges.",
#             "Harvesting": "Oranges are typically ready for harvest 7-12 months after flowering, depending on the variety. Indicators include a change in color, firmness, and sweetness. Use sharp pruning shears to cut oranges from the tree, leaving a short stem attached to the fruit.",
#             "Post-Harvest Management": "Handle harvested oranges gently to avoid bruising. Store them in a cool, shaded area.",
#             "Storage Conditions": "Store oranges in a cool, dark place. They can be refrigerated to extend their shelf life.",
#             "Processing & Packaging": "If needed, oranges can be processed into juice, marmalade, or dried slices. Pack oranges in breathable containers to help maintain quality during storage and transport.",
#             "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or frost), and improper pruning techniques. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
#         },

#         {"name": "Papaya Cultivation Guide",
#             "Introduction": "Papayas (Carica papaya) are tropical fruit trees known for their sweet, juicy flesh and vibrant orange color. They thrive in warm climates and can produce fruit year-round under optimal conditions. This guide outlines the complete process for cultivating papayas, from planting to harvesting.",
#             "Materials Required": "- Quality papaya seeds or seedlings from reputable nurseries\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for effective moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
#             "Soil Preparation": "Papayas prefer well-drained, sandy loam or loamy soils with a pH of 6.0 to 6.5. Prepare the soil by tilling and incorporating organic matter to enhance drainage and fertility.",
#             "Plant Selection & Treatment": "Choose disease-resistant papaya varieties suited to your climate. If using seeds, soak them for a few hours before planting to improve germination rates.",
#             "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
#             "Planting Time": "The best time to plant papayas is in the spring when temperatures are consistently warm.",
#             "Spacing & Depth": "Space papaya plants 6-10 feet apart to allow for their large canopy and root system. Plant seeds or seedlings at a depth of about 0.5 to 1 inch.",
#             "Seeding/Transplanting Methods": "Direct Seeding: Plant seeds directly in the ground after the last frost.\nTransplanting: Start seedlings indoors and transplant them when they are about 12 inches tall.",
#             "Watering Requirements": "Water young papaya plants regularly, especially during dry spells. Papayas require consistent moisture but do not tolerate waterlogging.",
#             "Nutrient Management": "Apply a balanced fertilizer every 4-6 weeks during the growing season. Use organic compost to enhance soil fertility.",
#             "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
#             "Pest & Disease Management": "Monitor for pests such as aphids, whiteflies, and fruit flies. Manage diseases like powdery mildew and root rot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
#             "Special Care During Growth": "- Seedling Stage: Protect young plants from extreme weather and pests. Use row covers if necessary to shield from frost and insects.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune any dead or damaged leaves to promote healthy growth.\n- Fruit Development Stage: Ensure adequate water supply during fruit development. Thin excess fruits if necessary to allow for larger fruit size.",
#             "Harvesting": "Papayas are typically ready for harvest 6-12 months after planting, depending on the variety. Indicators include a change in skin color from green to yellow and a sweet aroma. Use a sharp knife to cut the fruit from the tree, leaving a small portion of the stem attached.",
#             "Post-Harvest Management": "Handle harvested papayas gently to avoid bruising. Store them in a cool, shaded area.",
#             "Storage Conditions": "Store papayas at room temperature to ripen further. Once ripe, they can be refrigerated for a short period to extend freshness.",
#             "Processing & Packaging": "If needed, papayas can be processed into smoothies, salads, or dried fruit. Pack papayas in breathable containers to maintain quality during storage and transport.",
#             "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or flooding), and improper watering practices. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
#         },

#         {"name": "Coffee Cultivation Guide",
#             "Introduction": "Coffee (Coffea spp.) is one of the most widely consumed beverages globally, known for its stimulating properties and rich flavor. It thrives in tropical climates, typically at higher altitudes, where conditions are ideal for its growth. This guide outlines the complete process for cultivating coffee, from planting to harvesting.",
#             "Materials Required": "- Quality coffee seedlings or seeds from reputable nurseries\n- Balanced fertilizers rich in nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for effective moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
#             "Soil Preparation": "Coffee prefers well-drained, loamy soils with a pH of 6.0 to 6.5. Prepare the soil by tilling and incorporating organic matter to enhance fertility and drainage.",
#             "Plant Selection & Treatment": "Choose disease-resistant coffee varieties suitable for your climate. If using seeds, soak them for 24 hours to improve germination rates.",
#             "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
#             "Planting Time": "The best time to plant coffee is at the beginning of the rainy season.",
#             "Spacing & Depth": "Space coffee plants 5-8 feet apart to allow for proper growth and air circulation. Plant seedlings at a depth that matches their nursery height, ensuring the root collar is level with the soil surface.",
#             "Seeding/Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, place the seedling in the hole, backfill gently, and water thoroughly after planting.",
#             "Watering Requirements": "Water young coffee plants regularly to establish roots, especially during dry spells. Mature plants prefer consistent moisture but should not be waterlogged.",
#             "Nutrient Management": "Apply a balanced fertilizer every 3-4 months during the growing season. Use organic compost to enhance soil fertility.",
#             "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
#             "Pest & Disease Management": "Monitor for pests such as coffee borer beetles and leaf rust. Manage diseases like root rot and leaf spot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
#             "Special Care During Growth": "- Seedling Stage: Protect young plants from extreme weather and pests. Use shade cloth if necessary to shield from intense sunlight.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to shape plants and remove any dead or diseased branches.\n- Flowering and Fruit Development Stage: Ensure adequate water supply during flowering and fruit set to maximize yield and fruit quality. Monitor for fruit fly infestations and control as necessary.",
#             "Harvesting": "Coffee cherries are typically ready for harvest 7-9 months after flowering, depending on the variety. Indicators include a change in color from green to bright red or yellow. Harvest coffee cherries by hand, picking only the ripe ones. Use a selective picking method for quality.",
#             "Post-Harvest Management": "Handle harvested cherries gently to avoid bruising. Process them as soon as possible to prevent spoilage.",
#             "Processing Methods": "Use either the dry method (sun-drying cherries) or the wet method (fermenting and washing cherries) to extract the coffee beans.",
#             "Storage Conditions": "Store processed coffee beans in a cool, dry place to prevent spoilage and maintain flavor.",
#             "Processing & Packaging": "Pack coffee beans in airtight containers to help preserve freshness during storage and transport.",
#             "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or frost), and fluctuating market prices. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
#         }

#     ]

# # Dropdown to select crop
# selected_crop = st.selectbox("Select a crop to view details:", [crop["name"] for crop in cropGuide])

# # Display selected crop details
# crop_details = next((crop for crop in cropGuide if crop["name"] == selected_crop), None)

# if crop_details:
#     st.subheader(f"{selected_crop} Cultivation Details")
#     for index, (key, value) in enumerate(crop_details.items()):
#         if key != "name":
#                 st.markdown(f"**{key}:** {value}")




# About Us Page - Team Members Section
elif app_mode == "ABOUT US":
    st.markdown("<h1 style='text-align: center; color: white;'>Team Members</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: white;'>Meet The Developers</h5>", unsafe_allow_html=True)

    # Team Member Data
    team_members = [
        {"name": "Srija Kontham", "linkedin": "#", "github": "#", "instagram": "#"},
        {"name": "Sowmya Sri Devagoni", "linkedin": "#", "github": "#", "instagram": "#"},
        {"name": "Sai Vishnu Teja Madhanambeti", "linkedin": "#", "github": "#", "instagram": "#"},
    ]

    # Display Team Cards
    cols = st.columns(3)  
    for index, member in enumerate(team_members):
        with cols[index % 3]:
            st.markdown(f"**    **")
            st.markdown(f"**{member['name']}**")
            st.markdown(f"**Masters | CSE**  \nCentral Michigan University")

#Prediction Page
elif(app_mode=="DISEASE DETECTION"):
    st.markdown("""
        <h1 style='text-align: center; color: green;'>🌿 DISEASE DETECTION 🌿</h1>
    """, unsafe_allow_html=True)
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))



# FARMING GUIDE Page 
elif(app_mode == "FARMING GUIDE"):
    st.markdown("""
         <h1 style='text-align: center; color: green;'>🌿 CROP FARMING GUIDE 🌿</h1>
    """, unsafe_allow_html=True)

    cropGuide = [
            {"name": "Maize Cultivation Guide", 
            "Introduction": "Maize (Zea mays), also known as corn, is a key cereal crop widely cultivated for its grains. This guide covers the complete process for cultivating maize from seed selection to harvesting.",
            "Materials Required": "- High-quality maize seeds (hybrid or improved varieties)\n- Fertilizers (Nitrogen, Phosphorus, Potassium)\n- Machinery (tractors, hand tools, seed planters)\n- Pest control (herbicides, insecticides)\n- Irrigation equipment (drip or furrow irrigation)",
            "Soil Preparation": "Maize thrives in well-drained loam soils with a pH of 5.8 to 7.0. Till the soil to improve aeration and break up clods.",
            "Seed Selection & Treatment": "Choose high-yielding, drought-resistant varieties. Treat seeds with fungicides or insecticides for protection.",
            "Field Preparation": "Level the field for even water distribution. Optimize row spacing for maximum sunlight exposure.",
            "Planting Time": "Typically planted at the beginning of the rainy season, between April and June, depending on the region.",
            "Spacing & Depth": "Plant seeds at 20-25 cm within rows and 60-75 cm between rows, at a depth of 2-5 cm.",
            "Seeding Methods": "- **Direct Seeding:** Plant seeds manually or with seed planters.",
            "Watering Requirements": "Requires regular watering, especially during silking and tasseling. Use irrigation if rain is insufficient.",
            "Nutrient Management": "Apply fertilizers in split doses: at planting, early growth, and tasseling stages.",
            "Weed Control": "Manual weeding, hoeing, or herbicides. First weeding at 15-20 days after planting, followed by another at 30-40 days.",
            "Pest & Disease Management": "Monitor for maize borers, armyworms, and aphids. Use pesticides and integrated pest management (IPM).",
            "Harvesting": "Harvest when maize ears mature and husks dry. Moisture content should be 20-25%. Use handpicking or mechanical harvesters.",
            "Post-Harvest Management": "Dry grains to 13-14% moisture. Shell, clean, and store properly.",
            "Storage Conditions": "Store in a cool, dry place with ventilation to prevent mold and pests.",
            "Processing": "If needed, dry and mill the maize for further use.",
            "Challenges & Solutions": "Common issues: weather variability, pests, and water scarcity. Solutions: IPM, soil moisture monitoring, and resilient varieties."
            },
            
            {"name": "Rice Cultivation Guide", 
                "Introduction": "Rice Oryza sativa is a staple food crop in many parts of the world. This guide covers the complete process of cultivating rice from seed selection to harvesting.",
                "Materials Required": "- High-quality seeds\n- Fertilizers (Nitrogen, Phosphorus, Potassium)\n- Irrigation system\n- Machinery (tractors, transplanting machines, sickles)\n- Pest control (herbicides, pesticides)", 
                "Soil Preparation": "Rice grows best in clay or clay-loam soils with pH 5.5 to 6.5. Till the soil and level the field for even water distribution.", 
                "Seed Selection & Treatment": "Use high-yielding, pest-resistant seeds. Treat them with fungicides or insecticides to prevent infestations.", 
                "Field Preparation": "Level the field and create bunds (raised edges) to retain water.", 
                "Planting Time": "Plant at the onset of the rainy season, usually from May to June depending on the region.", 
                "Spacing & Depth": "For transplanting, use 20x15 cm spacing. For direct seeding, plant 2-3 cm deep.",
                "Seeding Methods": "- **Direct Seeding:** Broadcasting seeds or planting in rows.\n- **Transplanting:** Grow in a nursery and transfer seedlings after 20-30 days.",
                "Watering Requirements": "Maintain 5-10 cm of water during growth. Reduce water at the grain ripening stage.",
                "Nutrient Management": "Apply fertilizers in split doses: at planting, during tillering, and at panicle initiation.",
                "Weed Control": "Use manual weeding or herbicides. Weed 15-20 days after transplanting, then again at 40 days.",
                "Pest & Disease Management": "Watch for pests like stem borers and leafhoppers. Use pesticides and integrated pest management (IPM) practices.",
                "Harvesting": "Harvest when grains turn golden-yellow and 80-90% of grains are mature. Use sickles for small farms or mechanical harvesters for efficiency.",
                "Post-Harvest Management": "Dry grains to 14% moisture, thresh, winnow, and store in a cool, dry place to prevent spoilage.",
                "Challenges & Solutions": "Common issues include adverse weather, pests, and water scarcity. Use IPM, monitor water levels, and diversify crop varieties to mitigate risks."
            },
            
            {"name": "Jute Cultivation Guide",
                "Introduction": "Jute is a fibrous crop mainly grown for its strong, natural fibers, widely used in textiles and packaging. This guide covers the complete process for cultivating jute from seed selection to harvesting.",
                "Materials Required": "- High-quality, certified jute seeds (Corchorus olitorius or Corchorus capsularis)\n- Organic compost, nitrogen, phosphorus, and potassium fertilizers\n- Hand tools or tractors for soil preparation\n- Herbicides and pesticides for pest control\n- Irrigation system for controlled watering",
                "Soil Preparation": "Jute grows best in loamy, sandy-loam soils with good drainage and a pH range of 6.0 to 7.5. Prepare the soil by plowing and leveling it to break up clods and ensure good seedbed preparation.",
                "Seed Selection & Treatment": "Choose high-yielding and disease-resistant seed varieties. Soak seeds in water for 24 hours before planting to encourage germination.",
                "Field Preparation": "Clear and level the field for uniform water distribution. Create small bunds around the field if flooding is expected.",
                "Planting Time": "Jute is usually planted with the arrival of the monsoon, typically between March and May.",
                "Spacing & Depth": "Sow seeds in rows with a spacing of 25-30 cm between rows. Plant seeds 1-2 cm deep for optimal germination.",
                "Seeding Methods": "- **Broadcasting:** Scatter seeds evenly over the field.\n- **Row Planting:** Sow seeds in rows, which facilitates weeding and other management activities.",
                "Watering Requirements": "Jute requires regular moisture; maintain adequate moisture, especially during the early growth phase. Avoid waterlogging by ensuring proper drainage, particularly after heavy rains.",
                "Nutrient Management": "Apply a basal dose of nitrogen, phosphorus, and potassium fertilizers at planting. Additional nitrogen can be applied after thinning, about 20-25 days after sowing.",
                "Weed Control": "Perform manual weeding or apply selective herbicides as needed, especially in the early stages. Conduct the first weeding 15-20 days after sowing, followed by another after 30-40 days.",
                "Pest & Disease Management": "Monitor for common pests like jute hairy caterpillars and aphids. Use pesticides or integrated pest management (IPM) practices to control pests and diseases like stem rot and anthracnose.",
                "Harvesting": "Harvest jute when the plants are 10-12 feet tall and the lower leaves start to yellow, typically 100-120 days after planting. Cut the plants close to the base using a sickle or knife. For best fiber quality, harvest before the plants begin to flower.",
                "Post-Harvest Management": "Bundle the harvested jute plants and submerge them in clean, slow-moving water for retting (fermentation process to loosen the fibers). Retting usually takes 10-15 days; check fiber separation regularly.",
                "Challenges & Solutions": "Common issues include water availability, pest infestations, and improper retting. Use efficient irrigation and pest control methods, and monitor water levels carefully during retting to ensure fiber quality."
            },

            {"name": "Cotton Cultivation Guide",
                "Introduction": "Cotton is a major fiber crop valued for its soft, fluffy fibers used in textiles. This guide covers the complete process for cultivating cotton from seed selection to harvesting.",
                "Materials Required": "- High-quality, certified cotton seeds (e.g., Bt cotton or other pest-resistant varieties)\n- Nitrogen, phosphorus, potassium, and micronutrient fertilizers\n- Drip or furrow irrigation system\n- Herbicides and pesticides for pest control\n- Plows, tractors, and sprayers for field preparation and maintenance",
                "Soil Preparation": "Cotton grows best in well-drained sandy-loam soils with a pH of 6.0 to 7.5. Prepare the field by deep plowing, followed by harrowing to break clods and smooth the surface.",
                "Seed Selection & Treatment": "Choose high-yielding, pest-resistant seed varieties. Treat seeds with fungicides or insecticides to protect against soil-borne diseases and early pest infestations.",
                "Field Preparation": "Create furrows or beds for planting, depending on irrigation method. Ensure good drainage to prevent waterlogging, which cotton is sensitive to.",
                "Planting Time": "Cotton is typically planted in spring, from March to May, depending on the region and temperature.",
                "Spacing & Depth": "Plant seeds 3-5 cm deep, with a spacing of 75-100 cm between rows and 25-30 cm between plants.",
                "Seeding Methods": "- **Direct Seeding:** Plant seeds directly into prepared furrows or beds using seed drills or by hand.",
                "Watering Requirements": "Cotton requires consistent moisture, especially during the flowering and boll formation stages. Use drip or furrow irrigation to maintain adequate soil moisture, particularly during dry spells.",
                "Nutrient Management": "Apply basal fertilizer with phosphorus and potassium at planting. Apply nitrogen in split doses: one-third at planting, one-third during vegetative growth, and one-third at flowering.",
                "Weed Control": "Use manual weeding, hoeing, or herbicides to control weeds, particularly during early growth stages. Perform weeding about 20-30 days after planting and again if necessary at 45 days.",
                "Pest & Disease Management": "Monitor for common pests like bollworms, aphids, and whiteflies. Use integrated pest management (IPM) practices, including biological controls, to minimize pesticide use.",
                "Harvesting": "Harvest cotton when the bolls are fully open and the fibers are fluffy, typically 150-180 days after planting. Manual harvesting involves picking mature bolls by hand, while large farms use cotton-picking machines.",
                "Post-Harvest Management": "Allow harvested cotton to dry in a shaded, ventilated area. Clean and gin the cotton to separate seeds from fiber. Store cotton fibers in a dry, well-ventilated place to avoid moisture-related damage.",
                "Challenges & Solutions": "Common issues include pest infestations, water availability, and soil nutrient depletion. Use drought-resistant varieties, implement efficient irrigation, and follow IPM practices to manage pests."
            },

            {"name": "Coconut Cultivation Guide",
                "Introduction": "The coconut palm (Cocos nucifera) is cultivated for its fruit, providing oil, milk, and fiber. This guide covers key steps from seed selection to harvesting.",
                "Materials Required": "- High-quality coconut seedlings (dwarf or tall varieties)\n- Organic manure, NPK fertilizers\n- Drip or basin irrigation\n- Pesticides or biocontrol agents\n- Hand tools or mechanical equipment",
                "Soil Preparation": "Coconuts thrive in well-drained sandy loam with pH 5.5-7.5. Dig 1 x 1 x 1 m pits, fill with soil, compost, and organic manure for strong root growth.",
                "Seed Selection & Treatment": "Use disease-resistant, high-yielding seedlings. Dwarf varieties allow easy harvesting, while tall varieties are drought-resistant.",
                "Field Preparation": "Clear weeds and debris, ensure proper drainage, and space pits as per variety needs.",
                "Planting Time": "Best planted at the rainy season’s onset to reduce irrigation needs; can be planted year-round with irrigation.",
                "Spacing & Depth": "Tall varieties: 7.5-9m apart; Dwarf: 6.5-7m. Ensure roots are well covered.",
                "Seeding Methods": "Place seedlings in pits with the collar just above ground level.",
                "Watering Requirements": "Water regularly for the first three years. Mature trees are drought-resistant but benefit from consistent irrigation.",
                "Nutrient Management": "Apply balanced fertilizers three times a year with micronutrients like magnesium and boron. Add organic manure annually.",
                "Weed Control": "Weed regularly, especially in early growth. Mulching helps retain moisture and suppress weeds.",
                "Pest & Disease Management": "Control pests like rhinoceros beetles and red palm weevils using pesticides or biocontrols. Manage root wilt and bud rot with fungicides and pruning.",
                "Harvesting": "Mature coconuts (12 months after flowering) turn brown. Harvest every 45-60 days using climbing tools or mechanical lifters.",
                "Post-Harvest Management": "Store in a dry, ventilated area. Process copra by sun-drying or mechanical drying. Pack dried coconuts securely for transport.",
                "Challenges & Solutions": "Drought, pests, and soil depletion can be managed with drip irrigation, pest management, and organic soil amendments."
            },

            {"name": "Chickpea Cultivation Guide",
                "Introduction": "Chickpea (Cicer arietinum) is a popular legume grown for its protein-rich seeds, widely used in food production. This guide covers the complete process for cultivating chickpeas from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant chickpea seeds (desi or kabuli types)\n- Phosphorus-based fertilizers; minimal nitrogen\n- Drip or sprinkler irrigation\n- Herbicides and pesticides\n- Plows, tractors, and sprayers",
                "Soil Preparation": "Chickpeas grow best in well-drained, loamy soils with a pH of 6.0-7.5. Plow and harrow the field for good root penetration.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant seeds. Treat with rhizobium bacteria for nitrogen fixation and fungicides to prevent diseases.",
                "Field Preparation": "Clear weeds and level the field. Space rows to allow air circulation and reduce disease risk.",
                "Planting Time": "Best planted in cool, dry seasons, typically October-November.",
                "Spacing & Depth": "Space plants 30-40 cm apart in rows 45-60 cm apart. Sow seeds 5-8 cm deep based on soil moisture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Chickpeas require minimal watering but benefit from irrigation during flowering and pod filling. Avoid waterlogging.",
                "Nutrient Management": "Apply phosphorus at planting. Use potassium and micronutrients as needed based on soil tests.",
                "Weed Control": "Weed early and regularly, either manually or with herbicides. First weeding at 20-30 days, second at 45-50 days if needed.",
                "Pest & Disease Management": "Monitor for pests like pod borers and aphids. Use integrated pest management (IPM) and biopesticides as needed.",
                "Special Care During Growth": "- Seedling stage: Protect from pests, maintain moderate moisture.\n- Vegetative stage: Maintain phosphorus levels.\n- Flowering & pod-filling: Ensure adequate moisture for optimal yield.",
                "Harvesting": "Chickpeas mature in 3-4 months. Harvest when plants yellow and pods dry. Cut by hand for small farms; use combine harvesters for large-scale farming.",
                "Post-Harvest Management": "Sun-dry seeds to reduce moisture, thresh, and clean before storage or sale.",
                "Storage Conditions": "Store in dry, cool places with ventilation to prevent insect infestations and spoilage.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
                "Challenges & Solutions": "Common issues include pests, diseases, water stress, and nutrient deficiencies. Use IPM, resistant varieties, and soil testing to mitigate risks."
            },

            {"name": "Pigeon Pea Cultivation Guide",
                "Introduction": "Pigeon peas (Cajanus cajan) are a drought-resistant legume valued for their high protein content and use in various dishes. This guide covers the complete process for cultivating pigeon peas from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant pigeon pea seeds (early, medium, or late-maturing varieties)\n- Nitrogen, phosphorus, and potassium fertilizers; minimal nitrogen needed\n- Drip or furrow irrigation equipment\n- Herbicides and pesticides specific to pigeon pea pests\n- Hand tools or tractors for soil preparation, planting, and weeding",
                "Soil Preparation": "Pigeon peas grow best in well-drained sandy loam to clay loam soils with a pH of 6.0-7.5. Plow and harrow the field to create a fine seedbed.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suitable for your region. Treat seeds with fungicides to prevent seed-borne diseases.",
                "Field Preparation": "Clear the field of weeds and debris, ensuring good drainage.",
                "Planting Time": "Typically planted at the beginning of the rainy season or during the dry season in subtropical regions.",
                "Spacing & Depth": "Space plants 30-40 cm apart in rows 60-75 cm apart. Sow seeds 3-5 cm deep, depending on soil moisture and texture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Pigeon peas are drought-resistant but require adequate moisture during flowering and pod development. Irrigation may be necessary, especially in the first 60 days.",
                "Nutrient Management": "Apply phosphorus and potassium at planting and top-dress with nitrogen if necessary. Organic amendments can improve soil fertility.",
                "Weed Control": "Control weeds during early growth stages using manual weeding or herbicides. Mulching can help suppress weeds and retain soil moisture.",
                "Pest & Disease Management": "Monitor for pests such as pod borers, aphids, and whiteflies. Implement integrated pest management (IPM) strategies, including biological controls and chemical pesticides as needed.",
                "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain soil moisture.\n- Vegetative stage: Ensure adequate nutrients for strong growth.\n- Flowering & pod-filling: Maintain consistent moisture to maximize yield and seed quality.",
                "Harvesting": "Pigeon peas mature in 4-6 months. Harvest when pods are mature and dry. Cut by hand for small farms or use combine harvesters for large-scale farming.",
                "Post-Harvest Management": "Allow harvested plants to sun-dry before threshing to reduce seed moisture content.",
                "Storage Conditions": "Store pigeon peas in a dry, cool, and well-ventilated area to prevent spoilage and insect infestations.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags or containers.",
                "Challenges & Solutions": "Common issues include pest infestations, diseases, water stress, and nutrient deficiencies. Use disease-resistant varieties, practice crop rotation, and apply IPM strategies to manage risks."
            },

            {"name": "Moth Bean Cultivation Guide",
                "Introduction": "Moth beans (Vigna aconitifolia) are a drought-resistant legume commonly grown in arid regions. They are valued for their high protein content and culinary applications. This guide covers the complete process for cultivating moth beans from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant moth bean seeds\n- Phosphorus and potassium fertilizers; minimal nitrogen\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
                "Soil Preparation": "Moth beans thrive in well-drained sandy loam or clay soils with a pH of 6.0-8.0. Prepare the field by plowing and harrowing for a fine seedbed.",
                "Seed Selection & Treatment": "Choose high-yielding, drought-tolerant varieties. Treat seeds with fungicides or insecticides to prevent seed-borne diseases.",
                "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
                "Planting Time": "Typically planted at the onset of the monsoon season, between June and July.",
                "Spacing & Depth": "Space plants 30-45 cm apart in rows 60-75 cm apart. Sow seeds 3-5 cm deep based on soil moisture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Moth beans are drought-resistant but benefit from consistent moisture during flowering and pod development. Water if rainfall is insufficient.",
                "Nutrient Management": "Apply phosphorus and potassium at planting. Use nitrogen only if soil tests indicate a deficiency. Organic amendments improve soil fertility.",
                "Weed Control": "Control weeds early with manual weeding or herbicides. Mulching helps suppress weeds and retain soil moisture.",
                "Pest & Disease Management": "Monitor for pests like aphids, pod borers, and leafhoppers. Use integrated pest management (IPM) strategies as needed.",
                "Special Care During Growth": "- Seedling stage: Maintain moderate moisture and protect from pests.\n- Vegetative stage: Ensure adequate nutrients.\n- Flowering & pod-filling: Maintain moisture for optimal yield.",
                "Harvesting": "Harvest when pods mature and dry, typically 90-120 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
                "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
                "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
                "Challenges & Solutions": "Common issues include pests, diseases, and adverse weather. Use drought-resistant varieties, IPM practices, and proper soil management to mitigate risks."
            },

            {"name": "Mung Bean Cultivation Guide",
                "Introduction": "Mung beans (Vigna radiata) are small, green legumes highly valued for their nutritional content and culinary versatility. This guide covers the complete process for cultivating mung beans from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant mung bean seeds\n- Nitrogen, phosphorus, and potassium fertilizers (minimal nitrogen needed)\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
                "Soil Preparation": "Mung beans prefer well-drained sandy loam to loamy soils with a pH of 6.0-7.5. Prepare the field by plowing and harrowing to achieve a fine seedbed.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suitable for your climate. Treat seeds with fungicides to protect against soil-borne diseases.",
                "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
                "Planting Time": "Typically planted at the beginning of the rainy season or in warm, dry conditions between April and June.",
                "Spacing & Depth": "Space plants 30-40 cm apart in rows 45-60 cm apart. Sow seeds 2-4 cm deep based on soil moisture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Mung beans require adequate moisture, particularly during germination and flowering. Water if rainfall is insufficient, ensuring not to overwater to prevent root rot.",
                "Nutrient Management": "Apply phosphorus and potassium at planting. Additional nitrogen may be applied if needed, but usually, the natural fixation suffices. Incorporate organic matter to improve soil fertility.",
                "Weed Control": "Control weeds early through manual weeding or herbicides. Mulching helps suppress weeds and conserve soil moisture.",
                "Pest & Disease Management": "Monitor for pests like aphids, beetles, and thrips. Use integrated pest management (IPM) strategies as needed.",
                "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain adequate moisture.\n- Vegetative stage: Ensure sufficient nutrients for strong growth.\n- Flowering & pod-filling: Maintain moisture for optimal yield and quality.",
                "Harvesting": "Harvest when pods mature and dry, typically 60-90 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
                "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
                "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
                "Challenges & Solutions": "Common issues include pests, diseases, and adverse weather. Use disease-resistant varieties, IPM practices, and proper soil and water management to mitigate risks."
            },

            {"name": "Black Gram Cultivation Guide",
                "Introduction": "Black gram (Vigna mungo) is a highly nutritious legume valued for its high protein content and is widely used in various culinary dishes. This guide covers the complete process for cultivating black gram from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant black gram seeds\n- Phosphorus and potassium fertilizers (minimal nitrogen needed)\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
                "Soil Preparation": "Black gram prefers well-drained sandy loam to clay loam soils with a pH of 6.0-7.5. Prepare the field by plowing and harrowing to create a fine seedbed.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suitable for your climate. Treat seeds with fungicides or insecticides to protect against soil-borne diseases.",
                "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
                "Planting Time": "Typically planted at the beginning of the monsoon season or during warm, dry conditions between June and July.",
                "Spacing & Depth": "Space plants 30-45 cm apart in rows 60-75 cm apart. Sow seeds 3-5 cm deep based on soil moisture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Black gram requires adequate moisture, particularly during germination and flowering. Water if rainfall is insufficient, ensuring not to overwater to prevent root rot.",
                "Nutrient Management": "Apply phosphorus and potassium at planting. Additional nitrogen is generally not necessary due to nitrogen fixation. Incorporate organic matter to improve soil fertility.",
                "Weed Control": "Control weeds early through manual weeding or herbicides. Mulching helps suppress weeds and conserve soil moisture.",
                "Pest & Disease Management": "Monitor for pests like aphids, pod borers, and thrips. Use integrated pest management (IPM) strategies as needed.",
                "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain adequate moisture.\n- Vegetative stage: Ensure sufficient nutrients for strong growth.\n- Flowering & pod-filling: Maintain moisture for optimal yield and quality.",
                "Harvesting": "Harvest when pods mature and dry, typically 60-90 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
                "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
                "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
                "Challenges & Solutions": "Common issues include pests, diseases, and adverse weather. Use disease-resistant varieties, IPM practices, and proper soil and water management to mitigate risks."
            },

            {"name": "Lentil Cultivation Guide",
                "Introduction": "Lentils (Lens culinaris) are nutritious legumes known for their high protein and fiber content. They are widely cultivated for food and are a staple in many cuisines. This guide covers the complete process for cultivating lentils from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant lentil seeds\n- Phosphorus and potassium fertilizers (minimal nitrogen needed)\n- Drip or furrow irrigation\n- Herbicides and pesticides\n- Hand tools or tractors",
                "Soil Preparation": "Lentils prefer well-drained loamy or sandy soils with a pH of 6.0-7.5. Prepare the field by plowing and harrowing to create a fine seedbed.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant varieties suited to your region. Treat seeds with fungicides or insecticides to protect against seed-borne diseases.",
                "Field Preparation": "Clear the field of weeds and debris to ensure good seed-to-soil contact.",
                "Planting Time": "Lentils are typically planted in early spring or late winter, depending on the climate, when soil temperatures reach around 10-15°C (50-59°F).",
                "Spacing & Depth": "Space plants 25-30 cm apart in rows 45-60 cm apart. Sow seeds 2-3 cm deep based on soil moisture.",
                "Seeding Methods": "Direct seeding using seed drills or manual planting.",
                "Watering Requirements": "Lentils are drought-tolerant but need adequate moisture during germination and pod development. Water if rainfall is insufficient, particularly during flowering and seed filling.",
                "Nutrient Management": "Apply phosphorus and potassium at planting. Additional nitrogen is typically not needed due to nitrogen fixation. Incorporate organic matter to enhance soil fertility.",
                "Weed Control": "Control weeds during early growth using manual weeding or herbicides. Mulching can also help suppress weeds and retain soil moisture.",
                "Pest & Disease Management": "Monitor for pests such as aphids, lygus bugs, and root rots. Implement integrated pest management (IPM) strategies as needed.",
                "Special Care During Growth": "- Seedling stage: Protect young seedlings from pests and maintain adequate moisture.\n- Vegetative stage: Ensure sufficient nutrients for strong growth.\n- Flowering & pod-filling: Maintain moisture for optimal yield and quality.",
                "Harvesting": "Harvest when pods turn brown and dry, typically 80-100 days after planting. Hand-harvest for small farms; use combine harvesters for large-scale operations.",
                "Post-Harvest Management": "Sun-dry plants before threshing to reduce moisture content.",
                "Storage Conditions": "Store in dry, cool places with ventilation to prevent spoilage and insect infestations.",
                "Processing & Packaging": "Clean and grade seeds before packaging in breathable bags.",
                "Challenges & Solutions": "Common issues include pests, diseases, and variable weather. Use disease-resistant varieties, IPM practices, and proper soil and water management to mitigate risks."
            },

            {"name": "Pomegranate Cultivation Guide",
                "Introduction": "Pomegranates (Punica granatum) are nutritious fruits known for their health benefits and vibrant flavor. They are cultivated in many parts of the world and thrive in warm climates. This guide covers the complete process for cultivating pomegranates from planting to harvesting.",
                "Materials Required": "- High-quality pomegranate seeds or healthy seedlings from reputable nurseries\n- Balanced fertilizers with nitrogen, phosphorus, and potassium\n- Drip irrigation systems or furrow irrigation\n- Insecticides and fungicides for pest and disease management\n- Hand tools or tractors for planting, pruning, and maintenance",
                "Soil Preparation": "Pomegranates prefer well-drained, sandy loam to loamy soils with a pH of 5.5 to 7.0. Prepare the planting site by plowing and incorporating organic matter.",
                "Seed Selection & Treatment": "Choose disease-resistant varieties suitable for your region's climate. If using seeds, soak them overnight in water before planting to improve germination rates.",
                "Field Preparation": "Clear the site of weeds, rocks, and debris to ensure a clean planting environment.",
                "Planting Time": "Pomegranates are typically planted in spring after the last frost.",
                "Spacing & Depth": "Space plants 5-8 feet apart to allow for proper growth and air circulation. Plant seeds or seedlings at a depth of 1-2 inches, ensuring good soil contact.",
                "Seeding Methods": "Direct Seeding: Sow seeds directly into the prepared site. Transplanting: For seedlings, dig a hole slightly larger than the root ball and backfill with soil.",
                "Watering Requirements": "Pomegranates require regular watering, especially during the establishment phase; once established, they are drought-tolerant. Water deeply but infrequently to encourage deep root growth.",
                "Nutrient Management": "Apply a balanced fertilizer during the growing season, typically in early spring and again in late summer. Incorporate organic compost to improve soil fertility.",
                "Weed Control": "Control weeds using mulching and manual weeding to reduce competition for nutrients.",
                "Pest & Disease Management": "Monitor for pests such as aphids, whiteflies, and pomegranate butterflies. Implement integrated pest management (IPM) strategies, including the use of natural predators and organic pesticides.",
                "Special Care During Growth": "- Seedling stage: Protect young plants from extreme weather and pests. Use mulch to retain moisture.\n- Vegetative stage: Regularly check for nutrient deficiencies and pest infestations; apply fertilizers as needed.\n- Flowering & fruit development: Ensure adequate water during flowering and fruit set to promote healthy development.",
                "Harvesting": "Pomegranates are typically ready for harvest 5-7 months after flowering, when the fruit has a deep color and makes a metallic sound when tapped. Use sharp pruning shears to cut the fruit from the tree, avoiding damage to the branches and other fruit.",
                "Post-Harvest Management": "Handle fruits gently to prevent bruising; store in a cool, dry place.",
                "Storage Conditions": "Store pomegranates in a cool, dry environment; they can last several weeks to months in proper conditions.",
                "Processing & Packaging": "Clean and sort harvested fruits, discarding any damaged or rotten ones. Pack fruits in breathable containers to maintain quality during storage.",
                "Challenges & Solutions": "Common issues include susceptibility to pests, diseases, and environmental stresses such as drought or excessive moisture. Use disease-resistant varieties, implement proper irrigation practices, and monitor pest populations to mitigate challenges."
            },

            {"name": "Kidney Bean Cultivation Guide",
                "Introduction": "Kidney beans (Phaseolus vulgaris) are a high-protein legume commonly used in various cuisines. This guide covers the complete process for cultivating kidney beans from seed selection to harvesting.",
                "Materials Required": "- High-quality, disease-resistant kidney bean seeds\n- Phosphorus and potassium fertilizers; minimal nitrogen as beans fix their own nitrogen\n- Drip or sprinkler irrigation\n- Herbicides and pesticides for common kidney bean pests\n- Hand tools or tractors for soil preparation, planting, and weeding",
                "Soil Preparation": "Kidney beans thrive in well-drained, loamy soils with a pH between 6.0 and 7.0. Prepare the field by plowing and harrowing to create a fine tilth for easy root penetration.",
                "Seed Selection & Treatment": "Choose high-yielding, disease-resistant seed varieties. Treat seeds with fungicides or insecticides to protect against early soil-borne diseases and pests.",
                "Field Preparation": "Clear the field of weeds and debris, then level it. Mark rows with adequate spacing for air circulation and sunlight penetration.",
                "Planting Time": "Kidney beans are typically planted in spring when soil temperatures reach 15°C (59°F) and there is no risk of frost.",
                "Spacing & Depth": "Plant seeds 3-5 cm deep, with 8-10 cm between plants and 45-60 cm between rows.",
                "Seeding Methods": "Direct Seeding: Sow seeds directly into the field by hand or using a seed drill.",
                "Watering Requirements": "Kidney beans need regular watering, particularly during flowering and pod development. Avoid overwatering, as beans are sensitive to waterlogging.",
                "Nutrient Management": "Apply phosphorus and potassium at planting. Limit nitrogen since kidney beans fix atmospheric nitrogen. Supplement micronutrients if soil tests indicate deficiencies.",
                "Weed Control": "Weed control is essential, particularly in the early stages. Use manual weeding or herbicides as needed. Mulching around plants can help retain moisture and suppress weeds.",
                "Pest & Disease Management": "Monitor for pests like aphids, leafhoppers, and bean beetles. Use integrated pest management (IPM) practices and apply pesticides if necessary. Prevent diseases like root rot and blight by practicing crop rotation and avoiding waterlogged soil.",
                "Special Care During Growth": "- Seedling stage: Ensure moderate soil moisture and protect seedlings from pests.\n- Vegetative stage: Maintain nutrient levels to support robust leaf and stem growth.\n- Flowering & pod-filling stage: Provide consistent moisture during pod development to enhance yield and seed quality.",
                "Harvesting": "Harvest kidney beans when the pods are fully mature and dry, usually 90-120 days after planting. For small farms, harvest by hand by pulling up the entire plant. For larger farms, use a combine harvester to gather beans efficiently.",
                "Post-Harvest Management": "Allow the harvested plants to dry in the sun to reduce moisture in the seeds. Thresh the beans to separate them from the pods, then clean the seeds.",
                "Storage Conditions": "Store kidney beans in a dry, well-ventilated place to prevent mold and insect infestations.",
                "Processing & Packaging": "Clean and grade the beans for quality assurance before packaging. Pack beans in breathable bags or containers to maintain quality during storage.",
                "Challenges & Solutions": "Common issues include susceptibility to pests, diseases, and nutrient imbalances. Use disease-resistant seeds, monitor soil health, and apply IPM practices to control pests and diseases effectively."
            },

            {"name": "Banana Cultivation Guide",
                "Introduction": "Bananas (Musa spp.) are tropical fruits renowned for their sweet flavor and nutritional benefits. They thrive in warm, humid climates and are cultivated worldwide for both commercial and home production. This guide outlines the complete process for cultivating bananas, from planting to harvesting.",
                "Materials Required": "- Healthy banana suckers or tissue-cultured plantlets\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic matter such as compost\n- Drip or sprinkler irrigation systems for adequate moisture management\n- Insecticides and fungicides to manage pests and diseases\n- Hand tools (shovels, pruners) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Bananas prefer well-drained, rich loamy soils with a pH of 5.5 to 7.0. Prepare the soil by plowing and incorporating organic matter to improve fertility and drainage.",
                "Plant Selection & Treatment": "Select disease-free suckers from healthy parent plants or obtain tissue-cultured plantlets from a reputable source. If using suckers, cut them from the parent plant with a clean knife to avoid contamination.",
                "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The ideal time to plant bananas is at the beginning of the rainy season or during the warmer months.",
                "Spacing & Depth": "Space plants 8-10 feet apart in rows that are 10-12 feet apart to allow for proper growth and air circulation. Plant suckers or plantlets at the same depth they were growing in the nursery.",
                "Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots and backfill gently to avoid air pockets.",
                "Watering Requirements": "Bananas require consistent moisture; irrigate regularly, especially during dry spells. Aim for 1-2 inches of water per week.",
                "Nutrient Management": "Apply a balanced fertilizer in early spring and again mid-season. Add compost or organic mulch to enhance soil fertility.",
                "Weed Control": "Control weeds using mulching, which also helps retain soil moisture, and manual weeding to reduce competition for nutrients.",
                "Pest & Disease Management": "Monitor for pests such as banana weevils and aphids. Manage diseases like Panama disease and leaf spot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of biological pest control methods.",
                "Special Care During Growth": "- Seedling stage: Protect young plants from extreme weather and pests; consider using shade cloth if necessary.\n- Vegetative stage: Regularly check for nutrient deficiencies, especially potassium and magnesium, and address them promptly.\n- Flowering & fruit development stage: Ensure adequate water supply during flowering and fruit development to support healthy fruit formation.",
                "Harvesting": "Bananas are typically ready for harvest 9-12 months after planting, depending on the variety and growing conditions. Harvest when the fruit is plump, green, and the angle between the fruit and the stalk becomes more pronounced. Use a sharp knife or machete to cut the entire bunch from the plant. Handle the fruit carefully to avoid bruising.",
                "Post-Harvest Management": "Remove any excess leaves and handle harvested bananas gently to prevent damage. Store them in a cool, shaded area.",
                "Storage Conditions": "Store bananas at room temperature until they ripen. Avoid exposure to direct sunlight or excessive heat.",
                "Processing & Packaging": "If needed, bananas can be processed into products like banana chips or puree. Pack bananas in breathable boxes to allow for airflow and reduce spoilage during transport.",
                "Challenges & Solutions": "Common issues include susceptibility to pests and diseases, environmental stresses, and improper watering. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Banana Cultivation Guide",
                "Introduction": "Bananas (Musa spp.) are tropical fruits renowned for their sweet flavor and nutritional benefits. They thrive in warm, humid climates and are cultivated worldwide for both commercial and home production. This guide outlines the complete process for cultivating bananas, from planting to harvesting.",
                "Materials Required": "- Healthy banana suckers or tissue-cultured plantlets\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic matter such as compost\n- Drip or sprinkler irrigation systems for adequate moisture management\n- Insecticides and fungicides to manage pests and diseases\n- Hand tools (shovels, pruners) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Bananas prefer well-drained, rich loamy soils with a pH of 5.5 to 7.0. Prepare the soil by plowing and incorporating organic matter to improve fertility and drainage.",
                "Plant Selection & Treatment": "Select disease-free suckers from healthy parent plants or obtain tissue-cultured plantlets from a reputable source. If using suckers, cut them from the parent plant with a clean knife to avoid contamination.",
                "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The ideal time to plant bananas is at the beginning of the rainy season or during the warmer months.",
                "Spacing & Depth": "Space plants 8-10 feet apart in rows that are 10-12 feet apart to allow for proper growth and air circulation. Plant suckers or plantlets at the same depth they were growing in the nursery.",
                "Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots and backfill gently to avoid air pockets.",
                "Watering Requirements": "Bananas require consistent moisture; irrigate regularly, especially during dry spells. Aim for 1-2 inches of water per week.",
                "Nutrient Management": "Apply a balanced fertilizer in early spring and again mid-season. Add compost or organic mulch to enhance soil fertility.",
                "Weed Control": "Control weeds using mulching, which also helps retain soil moisture, and manual weeding to reduce competition for nutrients.",
                "Pest & Disease Management": "Monitor for pests such as banana weevils and aphids. Manage diseases like Panama disease and leaf spot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of biological pest control methods.",
                "Special Care During Growth": "- Seedling stage: Protect young plants from extreme weather and pests; consider using shade cloth if necessary.\n- Vegetative stage: Regularly check for nutrient deficiencies, especially potassium and magnesium, and address them promptly.\n- Flowering & fruit development stage: Ensure adequate water supply during flowering and fruit development to support healthy fruit formation.",
                "Harvesting": "Bananas are typically ready for harvest 9-12 months after planting, depending on the variety and growing conditions. Harvest when the fruit is plump, green, and the angle between the fruit and the stalk becomes more pronounced. Use a sharp knife or machete to cut the entire bunch from the plant. Handle the fruit carefully to avoid bruising.",
                "Post-Harvest Management": "Remove any excess leaves and handle harvested bananas gently to prevent damage. Store them in a cool, shaded area.",
                "Storage Conditions": "Store bananas at room temperature until they ripen. Avoid exposure to direct sunlight or excessive heat.",
                "Processing & Packaging": "If needed, bananas can be processed into products like banana chips or puree. Pack bananas in breathable boxes to allow for airflow and reduce spoilage during transport.",
                "Challenges & Solutions": "Common issues include susceptibility to pests and diseases, environmental stresses, and improper watering. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },


            {"name": "Grape Cultivation Guide",
                "Introduction": "Grapes (Vitis vinifera and other species) are versatile fruits used for fresh eating, drying (raisins), and wine production. They thrive in temperate climates and require specific growing conditions to produce high-quality fruit. This guide outlines the complete process for cultivating grapes, from planting to harvesting.",
                "Materials Required": "- Quality grapevines, either bare-root or potted, from reputable nurseries\n- Balanced fertilizers containing nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems for efficient moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (pruners, shovels) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Grapes prefer well-drained, sandy loam or clay loam soils with a pH of 6.0 to 6.8. Prepare the soil by tilling and incorporating organic matter to enhance fertility and drainage.",
                "Plant Selection & Treatment": "Select disease-resistant grape varieties suitable for your climate and purpose (table grapes, wine grapes, etc.). Inspect vines for signs of disease or damage before planting.",
                "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The ideal time to plant grapes is in early spring after the last frost or in the fall before the ground freezes.",
                "Spacing & Depth": "Space vines 6-10 feet apart in rows that are 8-10 feet apart to allow for proper air circulation and growth. Plant vines at the same depth they were growing in the nursery.",
                "Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, backfill gently, and water thoroughly after planting.",
                "Watering Requirements": "Grapes require regular watering during the first year to establish roots. Once established, they are drought-tolerant but still benefit from supplemental irrigation during dry spells, especially during fruit development.",
                "Nutrient Management": "Apply a balanced fertilizer in early spring and again mid-season. Use organic compost to improve soil health.",
                "Weed Control": "Control weeds through mulching, hand weeding, or the use of herbicides to reduce competition for nutrients and moisture.",
                "Pest & Disease Management": "Monitor for pests such as grapevine moths, aphids, and spider mites. Manage diseases like powdery mildew and downy mildew with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and natural predators.",
                "Special Care During Growth": "- Young Vine Stage: Protect young vines from extreme weather and pests; use support stakes or trellises to help young plants grow upward.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to encourage a strong structure and air circulation.\n- Flowering & Fruit Development Stage: Ensure consistent moisture during flowering and fruit set to maximize yield and fruit quality. Thin clusters if necessary to promote larger fruit size.",
                "Harvesting": "Grapes are typically ready for harvest 4-6 months after flowering, depending on the variety. They should be harvested when fully ripe, showing deep color and sweet flavor. Use sharp pruning shears to cut clusters from the vine. Handle the fruit carefully to avoid bruising.",
                "Post-Harvest Management": "Remove any damaged or rotten grapes and store them in a cool, shaded area.",
                "Storage Conditions": "Store grapes in a cool, dry place. Refrigeration can extend their shelf life, but they should be kept in breathable containers.",
                "Processing & Packaging": "If needed, grapes can be processed into products like grape juice, jelly, or wine. Pack grapes in breathable containers to allow airflow and reduce spoilage during transport.",
                "Challenges & Solutions": "Common issues include susceptibility to pests and diseases, climate-related issues, and improper watering. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Muskmelon Cultivation Guide",
                "Introduction": "Muskmelons (Cucumis melo var. cantaloupe) are sweet, aromatic fruits known for their juicy flesh and distinctive netted skin. They thrive in warm climates and are popular for their refreshing taste. This guide outlines the complete process for cultivating muskmelons, from planting to harvesting.",
                "Materials Required": "- Quality muskmelon seeds or seedlings from reputable sources\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic compost\n- Drip or overhead irrigation systems for efficient moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, hoes, pruners) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Muskmelons prefer well-drained, sandy loam or loamy soils with a pH of 6.0 to 6.8. Prepare the soil by tilling and mixing in organic matter to enhance drainage and fertility.",
                "Plant Selection & Treatment": "Choose disease-resistant varieties suited for your climate and market. If using seeds, soak them in water for a few hours before planting to improve germination rates.",
                "Field Preparation": "Clear the planting site of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The ideal time to plant muskmelons is after the last frost date when soil temperatures are consistently above 70°F (21°C).",
                "Spacing & Depth": "Space muskmelon plants 3-4 feet apart in rows that are 6-8 feet apart to allow for sprawling vines. Plant seeds or seedlings at a depth of about 1 inch.",
                "Seeding/Transplanting Methods": "Direct Seeding: Plant seeds directly into the ground after the soil warms up. Transplanting: Start seedlings indoors and transplant them once they are strong enough.",
                "Watering Requirements": "Muskmelons need consistent moisture, especially during germination and fruit development. Aim for about 1-2 inches of water per week, adjusting for rainfall.",
                "Nutrient Management": "Apply a balanced fertilizer at planting and again when vines begin to run. Use organic compost or mulch to enhance soil health.",
                "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
                "Pest & Disease Management": "Monitor for pests such as aphids, cucumber beetles, and spider mites. Manage diseases like powdery mildew and downy mildew with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of biological controls.",
                "Special Care During Growth": "- Seedling Stage: Protect young plants from pests and extreme weather. Use row covers if necessary to protect against pests and frost.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Support vines if necessary, especially when fruit begins to develop.\n- Fruit Development Stage: Ensure adequate water supply during fruit development to promote healthy growth and sweetness. Avoid watering directly on the fruit to prevent rot.",
                "Harvesting": "Muskmelons are typically ready for harvest 70-90 days after planting. Indicators include a change in color from green to yellow at the blossom end and a sweet aroma. Use a sharp knife or pruning shears to cut the fruit from the vine, leaving a short stem attached to the melon.",
                "Post-Harvest Management": "Handle harvested muskmelons gently to avoid bruising. Store them in a cool, shaded area.",
                "Storage Conditions": "Store muskmelons at room temperature until they are fully ripe. Once ripe, they can be refrigerated for a short period to extend freshness.",
                "Processing & Packaging": "If needed, muskmelons can be processed into smoothies, sorbets, or fruit salads. Pack muskmelons in breathable containers to help maintain quality during storage and transport.",
                "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses such as drought or excessive moisture, and improper watering practices. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Apple Cultivation Guide",
                "Introduction": "Apples (Malus domestica) are one of the most popular fruits worldwide, appreciated for their taste, versatility, and nutritional value. They grow best in temperate climates and can be cultivated in various soil types. This guide outlines the complete process for cultivating apples, from planting to harvesting.",
                "Materials Required": "- Quality apple tree seedlings or grafted varieties from reputable nurseries\n- Balanced fertilizers containing nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for effective moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Apples prefer well-drained, loamy soils with a pH of 6.0 to 7.0. Prepare the soil by tilling and incorporating organic matter to improve fertility and drainage.",
                "Plant Selection & Treatment": "Choose disease-resistant apple varieties suited to your climate, considering factors such as fruit flavor and harvest time. Inspect seedlings for signs of disease or damage before planting.",
                "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The best time to plant apple trees is in the fall or early spring when the trees are dormant.",
                "Spacing & Depth": "Space dwarf varieties 4-6 feet apart and standard varieties 10-15 feet apart to allow for proper growth and air circulation. Plant trees at a depth that matches their nursery height, ensuring the graft union is above soil level.",
                "Seeding/Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, place the tree in the hole, backfill gently, and water thoroughly after planting.",
                "Watering Requirements": "Water young apple trees regularly to establish roots, especially during dry spells. Once established, they are drought-tolerant but benefit from deep watering during fruit development.",
                "Nutrient Management": "Apply a balanced fertilizer in early spring and again in mid-season. Use organic compost to enhance soil health.",
                "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
                "Pest & Disease Management": "Monitor for pests such as codling moths, aphids, and spider mites. Manage diseases like apple scab and powdery mildew with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
                "Special Care During Growth": "- Young Tree Stage: Protect young trees from extreme weather and pests; consider using tree guards to prevent animal damage.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to shape trees and encourage a strong structure.\n- Flowering and Fruit Development Stage: Ensure consistent moisture during flowering and fruit set to maximize yield and fruit quality. Thin fruit if necessary to promote larger apples.",
                "Harvesting": "Apples are typically ready for harvest 4-6 months after flowering, depending on the variety. Indicators include a change in color, firm texture, and ease of detachment from the tree. Use sharp pruning shears to cut apples from the tree, leaving a short stem attached to the fruit.",
                "Post-Harvest Management": "Handle harvested apples gently to avoid bruising. Store them in a cool, shaded area.",
                "Storage Conditions": "Store apples in a cool, dark place. They can be refrigerated to extend their shelf life.",
                "Processing & Packaging": "If needed, apples can be processed into applesauce, cider, or dried slices. Pack apples in breathable containers to help maintain quality during storage and transport.",
                "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or frost), and improper pruning techniques. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Orange Cultivation Guide",
                "Introduction": "Oranges (Citrus sinensis) are one of the most popular citrus fruits, valued for their sweet, juicy flesh and high vitamin C content. They thrive in warm, subtropical to tropical climates. This guide outlines the complete process for cultivating oranges, from planting to harvesting.",
                "Materials Required": "- Quality orange tree seedlings or grafted varieties from reputable nurseries\n- Citrus-specific fertilizers containing nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for efficient moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Oranges prefer well-drained, sandy loam or clay loam soils with a pH of 6.0 to 7.5. Prepare the soil by tilling and incorporating organic matter to improve fertility and drainage.",
                "Plant Selection & Treatment": "Choose disease-resistant orange varieties suited to your climate, considering factors such as fruit flavor and harvest time. Inspect seedlings for signs of disease or damage before planting.",
                "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The best time to plant orange trees is in the spring after the danger of frost has passed.",
                "Spacing & Depth": "Space trees 12-25 feet apart, depending on the rootstock and tree variety, to allow for proper growth and air circulation. Plant trees at a depth that matches their nursery height, ensuring the graft union is above soil level.",
                "Seeding/Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, place the tree in the hole, backfill gently, and water thoroughly after planting.",
                "Watering Requirements": "Water young orange trees regularly to establish roots, especially during dry spells. Mature trees require deep watering during dry periods.",
                "Nutrient Management": "Apply a citrus-specific fertilizer in early spring and again in mid-season. Use organic compost to enhance soil health.",
                "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
                "Pest & Disease Management": "Monitor for pests such as aphids, spider mites, and citrus leaf miners. Manage diseases like citrus canker and root rot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
                "Special Care During Growth": "- Young Tree Stage: Protect young trees from extreme weather and pests; consider using tree guards to prevent animal damage.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to shape trees and encourage a strong structure.\n- Flowering and Fruit Development Stage: Ensure consistent moisture during flowering and fruit set to maximize yield and fruit quality. Thin fruit if necessary to promote larger oranges.",
                "Harvesting": "Oranges are typically ready for harvest 7-12 months after flowering, depending on the variety. Indicators include a change in color, firmness, and sweetness. Use sharp pruning shears to cut oranges from the tree, leaving a short stem attached to the fruit.",
                "Post-Harvest Management": "Handle harvested oranges gently to avoid bruising. Store them in a cool, shaded area.",
                "Storage Conditions": "Store oranges in a cool, dark place. They can be refrigerated to extend their shelf life.",
                "Processing & Packaging": "If needed, oranges can be processed into juice, marmalade, or dried slices. Pack oranges in breathable containers to help maintain quality during storage and transport.",
                "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or frost), and improper pruning techniques. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Papaya Cultivation Guide",
                "Introduction": "Papayas (Carica papaya) are tropical fruit trees known for their sweet, juicy flesh and vibrant orange color. They thrive in warm climates and can produce fruit year-round under optimal conditions. This guide outlines the complete process for cultivating papayas, from planting to harvesting.",
                "Materials Required": "- Quality papaya seeds or seedlings from reputable nurseries\n- Balanced fertilizers with nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for effective moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Papayas prefer well-drained, sandy loam or loamy soils with a pH of 6.0 to 6.5. Prepare the soil by tilling and incorporating organic matter to enhance drainage and fertility.",
                "Plant Selection & Treatment": "Choose disease-resistant papaya varieties suited to your climate. If using seeds, soak them for a few hours before planting to improve germination rates.",
                "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The best time to plant papayas is in the spring when temperatures are consistently warm.",
                "Spacing & Depth": "Space papaya plants 6-10 feet apart to allow for their large canopy and root system. Plant seeds or seedlings at a depth of about 0.5 to 1 inch.",
                "Seeding/Transplanting Methods": "Direct Seeding: Plant seeds directly in the ground after the last frost.\nTransplanting: Start seedlings indoors and transplant them when they are about 12 inches tall.",
                "Watering Requirements": "Water young papaya plants regularly, especially during dry spells. Papayas require consistent moisture but do not tolerate waterlogging.",
                "Nutrient Management": "Apply a balanced fertilizer every 4-6 weeks during the growing season. Use organic compost to enhance soil fertility.",
                "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
                "Pest & Disease Management": "Monitor for pests such as aphids, whiteflies, and fruit flies. Manage diseases like powdery mildew and root rot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
                "Special Care During Growth": "- Seedling Stage: Protect young plants from extreme weather and pests. Use row covers if necessary to shield from frost and insects.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune any dead or damaged leaves to promote healthy growth.\n- Fruit Development Stage: Ensure adequate water supply during fruit development. Thin excess fruits if necessary to allow for larger fruit size.",
                "Harvesting": "Papayas are typically ready for harvest 6-12 months after planting, depending on the variety. Indicators include a change in skin color from green to yellow and a sweet aroma. Use a sharp knife to cut the fruit from the tree, leaving a small portion of the stem attached.",
                "Post-Harvest Management": "Handle harvested papayas gently to avoid bruising. Store them in a cool, shaded area.",
                "Storage Conditions": "Store papayas at room temperature to ripen further. Once ripe, they can be refrigerated for a short period to extend freshness.",
                "Processing & Packaging": "If needed, papayas can be processed into smoothies, salads, or dried fruit. Pack papayas in breathable containers to maintain quality during storage and transport.",
                "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or flooding), and improper watering practices. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            },

            {"name": "Coffee Cultivation Guide",
                "Introduction": "Coffee (Coffea spp.) is one of the most widely consumed beverages globally, known for its stimulating properties and rich flavor. It thrives in tropical climates, typically at higher altitudes, where conditions are ideal for its growth. This guide outlines the complete process for cultivating coffee, from planting to harvesting.",
                "Materials Required": "- Quality coffee seedlings or seeds from reputable nurseries\n- Balanced fertilizers rich in nitrogen, phosphorus, and potassium; organic compost\n- Drip irrigation systems or hoses for effective moisture management\n- Insecticides, fungicides, and organic pest management solutions\n- Hand tools (shovels, pruning shears, hoes) or tractors for planting, maintenance, and harvesting",
                "Soil Preparation": "Coffee prefers well-drained, loamy soils with a pH of 6.0 to 6.5. Prepare the soil by tilling and incorporating organic matter to enhance fertility and drainage.",
                "Plant Selection & Treatment": "Choose disease-resistant coffee varieties suitable for your climate. If using seeds, soak them for 24 hours to improve germination rates.",
                "Field Preparation": "Clear the planting area of weeds, stones, and debris to ensure a clean environment for planting.",
                "Planting Time": "The best time to plant coffee is at the beginning of the rainy season.",
                "Spacing & Depth": "Space coffee plants 5-8 feet apart to allow for proper growth and air circulation. Plant seedlings at a depth that matches their nursery height, ensuring the root collar is level with the soil surface.",
                "Seeding/Transplanting Methods": "Transplanting: Dig a hole large enough to accommodate the roots, place the seedling in the hole, backfill gently, and water thoroughly after planting.",
                "Watering Requirements": "Water young coffee plants regularly to establish roots, especially during dry spells. Mature plants prefer consistent moisture but should not be waterlogged.",
                "Nutrient Management": "Apply a balanced fertilizer every 3-4 months during the growing season. Use organic compost to enhance soil fertility.",
                "Weed Control": "Control weeds through mulching, which helps retain moisture and suppress weed growth, and manual weeding to reduce competition.",
                "Pest & Disease Management": "Monitor for pests such as coffee borer beetles and leaf rust. Manage diseases like root rot and leaf spot with proper sanitation and resistant varieties. Implement integrated pest management (IPM) strategies, including cultural controls and the use of beneficial insects.",
                "Special Care During Growth": "- Seedling Stage: Protect young plants from extreme weather and pests. Use shade cloth if necessary to shield from intense sunlight.\n- Vegetative Stage: Regularly check for nutrient deficiencies and address them promptly. Prune to shape plants and remove any dead or diseased branches.\n- Flowering and Fruit Development Stage: Ensure adequate water supply during flowering and fruit set to maximize yield and fruit quality. Monitor for fruit fly infestations and control as necessary.",
                "Harvesting": "Coffee cherries are typically ready for harvest 7-9 months after flowering, depending on the variety. Indicators include a change in color from green to bright red or yellow. Harvest coffee cherries by hand, picking only the ripe ones. Use a selective picking method for quality.",
                "Post-Harvest Management": "Handle harvested cherries gently to avoid bruising. Process them as soon as possible to prevent spoilage.",
                "Processing Methods": "Use either the dry method (sun-drying cherries) or the wet method (fermenting and washing cherries) to extract the coffee beans.",
                "Storage Conditions": "Store processed coffee beans in a cool, dry place to prevent spoilage and maintain flavor.",
                "Processing & Packaging": "Pack coffee beans in airtight containers to help preserve freshness during storage and transport.",
                "Challenges & Solutions": "Common challenges include susceptibility to pests and diseases, environmental stresses (such as drought or frost), and fluctuating market prices. Choose disease-resistant varieties, implement good cultural practices, and monitor environmental conditions to mitigate these challenges."
            }

        ]

    # Dropdown to select crop
    selected_crop = st.selectbox("Select a crop to view details:", [crop["name"] for crop in cropGuide])

    # Display selected crop details
    crop_details = next((crop for crop in cropGuide if crop["name"] == selected_crop), None)

    if crop_details:
        st.subheader(f"{selected_crop} Cultivation Details")
        for index, (key, value) in enumerate(crop_details.items()):
            if key != "name":
                    st.markdown(f"**{key}:** {value}")



