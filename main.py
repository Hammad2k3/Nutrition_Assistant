import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any
import json
from datetime import datetime
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()


# ========== CLIENT INITIALIZATION ==========
def init_groq_client():
    """Initialize and return Groq client with error handling"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("❌ GROQ_API_KEY not found in .env file")
            st.stop()

        return ChatGroq(
            temperature=0.3,  # Lower temperature for more deterministic JSON output
            model_name="llama3-70b-8192",
            api_key=api_key
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {str(e)}")
        st.stop()


# ========== PROMPT TEMPLATE ==========
def create_prompt_template() -> ChatPromptTemplate:
    """Create the structured prompt template with clear JSON examples"""
    return ChatPromptTemplate.from_template("""
    You are NutriAI, an advanced AI nutritionist. Generate a personalized {duration}-day diet plan in VALID JSON format.

    === USER PROFILE ===
    - Name: {name}
    - Age: {age}
    - Weight: {weight} kg
    - Height: {height} cm
    - Activity: {activity_level}
    - Goals: {goal} (primary), {secondary_goals} (secondary)
    - Preferences: {dietary_preferences}
    - Requirements: {dietary_requirements}
    - Restrictions: {restrictions}
    - Region: {region}
    - Cuisines: {preferred_cuisines}
    - Budget: {budget}
    - Meal Frequency: {meal_frequency}

    === REGIONAL CONSIDERATIONS ===
    Incorporate authentic and regionally appropriate foods based on the user's region ({region}). Focus on locally available ingredients and traditional cooking methods common in that area. Adjust spice levels, cooking techniques, and meal compositions to align with regional dietary patterns.

    === OUTPUT FORMAT ===
    Return ONLY a JSON object with no other text around it. The JSON must follow this structure:
    {{
        "metadata": {{
            "generated_at": "{timestamp}",
            "plan_duration": {duration}
        }},
        "user_profile": {{
            "bmi": 22.1,
            "bmr": 1650,
            "daily_calories": 2200,
            "macros": {{
                "protein": {{"percent": 30, "grams": 165}},
                "carbs": {{"percent": 40, "grams": 220}},
                "fats": {{"percent": 30, "grams": 73}}
            }},
            "micro_nutrients": ["iron", "vitamin D"],
            "considerations": []
        }},
        "daily_plans": {{
            "day_1": {{
                "meals": {{
                    "breakfast": {{
                        "name": "Meal Name",
                        "desc": "Description",
                        "ingredients": ["item1", "item2"],
                        "nutrition": {{
                            "calories": 350,
                            "protein": 20,
                            "carbs": 45,
                            "fats": 8,
                            "fiber": 5
                        }},
                        "prep": "Preparation steps",
                        "time": "08:00"
                    }}
                }},
                "snacks": {{
                    "morning_snack": {{
                        "name": "Snack Name",
                        "nutrition": {{
                            "calories": 150,
                            "protein": 5,
                            "carbs": 20,
                            "fats": 5
                        }}
                    }}
                }},
                "hydration": {{
                    "water": "2L minimum",
                    "other": ["herbal tea", "electrolyte drink"]
                }}
            }}
        }},
        "shopping_list": {{
            "proteins": ["chicken", "tofu"],
            "carbs": ["quinoa", "sweet potatoes"],
            "vegetables": ["spinach", "broccoli"],
            "fruits": ["berries", "bananas"],
            "other": ["olive oil", "nuts"]
        }},
        "recommendations": {{
            "general": "General advice",
            "supplements": ["vitamin D", "omega-3"],
            "tracking": "Progress tracking tips"
        }}
    }}

    IMPORTANT: Return ONLY the JSON - no explanation, no markdown code blocks, no additional text.
    """).partial(timestamp=datetime.now().isoformat())


# ========== JSON HANDLING ==========
def extract_and_parse_json(raw_output: str) -> Dict[str, Any]:
    """Extract and parse JSON from LLM output with robust error handling"""
    try:
        # Remove any non-JSON content (explanations, code blocks, etc.)
        json_pattern = r'({[\s\S]*})'
        match = re.search(json_pattern, raw_output)

        if match:
            json_str = match.group(1)
        else:
            json_str = raw_output

        # Remove markdown code block formatting
        json_str = re.sub(r'```(?:json)?\s*|\s*```', '', json_str).strip()

        # Fix common JSON syntax issues
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas

        # Parse the JSON
        return json.loads(json_str)

    except json.JSONDecodeError as e:
        st.error(f"🔴 JSON Parsing Error: {str(e)}")
        st.text_area("Problematic JSON Output", raw_output, height=200)
        return None
    except Exception as e:
        st.error(f"🔴 Error processing output: {str(e)}")
        return None


# ========== USER INPUT ==========
def get_user_input() -> Dict[str, Any]:
    """Collect and return user input through Streamlit form"""
    st.title("🍏 NutriAI")
    st.subheader("Your Advanced Nutrition Assistant")

    with st.form("user_input_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name", placeholder="Alex Johnson")
            age = st.number_input("Age", min_value=12, max_value=120, value=30)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
            activity_level = st.selectbox(
                "Activity Level",
                ["Sedentary (little to no exercise)",
                 "Lightly Active (light exercise 1-3 days/week)",
                 "Moderately Active (moderate exercise 3-5 days/week)",
                 "Very Active (hard exercise 6-7 days/week)",
                 "Extremely Active (very hard exercise & physical job)"]
            )
            dietary_preferences = st.multiselect(
                "Dietary Preferences",
                ["Vegetarian", "Vegan", "Pescatarian", "Flexitarian", "None"]
            )

        with col2:
            dietary_requirements = st.multiselect(
                "Dietary Requirements",
                ["Gluten-free", "Dairy-free", "Keto", "Paleo", "Low-FODMAP", "Halal", "Kosher", "None"]
            )
            restrictions = st.text_input("Food Restrictions/Allergies", placeholder="e.g., nuts, shellfish")
            goal = st.selectbox(
                "Primary Goal",
                ["Weight Loss", "Weight Gain", "Maintain Weight", "Muscle Building", "Improve Health"]
            )
            region = st.selectbox(
                "Your Region",
                ["North America", "South America", "Western Europe", "Eastern Europe",
                 "Mediterranean", "Middle East", "South Asia", "East Asia",
                 "Southeast Asia", "Africa", "Australia/Oceania"]
            )
            preferred_cuisines = st.multiselect(
                "Preferred Cuisines",
                ["Mediterranean", "Asian", "Indian", "Mexican", "American", "European",
                 "Middle Eastern", "African", "Caribbean", "Latin American"]
            )

        budget = st.select_slider(
            "Budget",
            options=["Low", "Medium", "High"]
        )
        meal_frequency = st.select_slider(
            "Meal Frequency",
            options=["3 meals", "3 meals + 1 snack", "3 meals + 2 snacks", "5-6 small meals"]
        )
        duration = st.selectbox(
            "Plan Duration (days)",
            [7, 14, 21, 28],
            index=0
        )

        if st.form_submit_button("Generate Nutrition Plan"):
            return {
                "name": name,
                "age": age,
                "weight": weight,
                "height": height,
                "activity_level": activity_level,
                "dietary_preferences": ", ".join(dietary_preferences) if dietary_preferences else "None",
                "dietary_requirements": ", ".join(dietary_requirements) if dietary_requirements else "None",
                "restrictions": restrictions if restrictions else "None",
                "goal": goal,
                "region": region,
                "preferred_cuisines": ", ".join(preferred_cuisines) if preferred_cuisines else "Any",
                "budget": budget,
                "meal_frequency": meal_frequency,
                "duration": duration,
                "secondary_goals": "None"  # Default value
            }
    return None


# ========== PLAN DISPLAY ==========
def display_plan(plan: Dict[str, Any]):
    """Display the generated nutrition plan"""
    st.success("✅ Your Personalized Nutrition Plan is Ready!")

    # Metadata and quick stats in a clean header area
    st.markdown("### 📊 Your Nutrition Dashboard")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Daily Calories", plan['user_profile']['daily_calories'])
    with col2:
        st.metric("Plan Duration", f"{plan['metadata']['plan_duration']} days")
    with col3:
        st.metric("BMI", f"{plan['user_profile'].get('bmi', 'N/A')}")
    with col4:
        st.metric("BMR", f"{plan['user_profile'].get('bmr', 'N/A')}")

    # Tabs for main sections
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Daily Meal Plan", "🥗 Nutrition Profile", "🛒 Shopping List", "💡 Tips & Advice"])

    with tab1:
        # Calendar-style day selection
        st.markdown("### Select Your Day")
        days = list(plan['daily_plans'].keys())

        # Create a row of buttons for day selection
        cols = st.columns(min(7, len(days)))
        selected_day = st.session_state.get('selected_day', days[0])

        # Create day buttons in a week format
        for i, day in enumerate(days):
            day_num = day.split('_')[-1] if '_' in day else day.replace('day_', '')
            # Use numbered days with a cleaner display
            with cols[i % len(cols)]:
                if st.button(f"Day {day_num}", key=f"day_btn_{day}",
                             use_container_width=True,
                             type="primary" if day == selected_day else "secondary"):
                    st.session_state.selected_day = day
                    selected_day = day

        day_plan = plan['daily_plans'][selected_day]

        # Display daily meal schedule on a timeline
        st.markdown(f"### Day {selected_day.split('_')[-1]} Meal Schedule")

        # Create visual timeline for meals
        meal_times = []
        if 'meals' in day_plan:
            for meal_type, meal in day_plan['meals'].items():
                time = meal.get('time', '')
                if time:
                    meal_times.append((time, meal_type.capitalize(), meal['name'], 'meal'))

        if 'snacks' in day_plan and day_plan['snacks']:
            for snack_type, snack in day_plan['snacks'].items():
                # Assign default times if not specified
                if snack_type == 'morning_snack':
                    time = '10:30'
                elif snack_type == 'afternoon_snack':
                    time = '15:30'
                elif snack_type == 'evening_snack':
                    time = '20:00'
                else:
                    time = '12:00'
                meal_times.append((time, snack_type.replace('_', ' ').title(), snack['name'], 'snack'))

        # Sort by time
        meal_times.sort(key=lambda x: x[0])

        # Display timeline
        if meal_times:
            for time, meal_type, name, meal_kind in meal_times:
                icon = "🍳" if meal_kind == 'meal' else "🍌"
                st.markdown(f"**{time}** - {icon} **{meal_type}**: {name}")

        # Meal containers with better visual hierarchy
        st.markdown("### Meals & Snacks")

        # Meals section
        if 'meals' in day_plan:
            for meal_type, meal in day_plan['meals'].items():
                with st.container():
                    st.markdown(f"#### 🍳 {meal_type.capitalize()}: {meal['name']}")

                    # Description as a highlighted quote
                    st.markdown(f"> {meal['desc']}")

                    # Two column layout for ingredients and nutrition
                    meal_col1, meal_col2 = st.columns([3, 2])

                    with meal_col1:
                        # Ingredients with checkboxes
                        st.markdown("**📋 Ingredients:**")
                        for item in meal['ingredients']:
                            st.checkbox(item, key=f"{selected_day}_{meal_type}_{item}", label_visibility="visible")

                        # Preparation with numbered steps for better readability
                        st.markdown("**👨‍🍳 Preparation:**")
                        prep_steps = meal['prep'].split('. ')
                        for i, step in enumerate(prep_steps):
                            if step:  # Check if step is not empty
                                step = step.strip()
                                if not step.endswith('.'):
                                    step += '.'
                                st.markdown(f"{i + 1}. {step}")

                    with meal_col2:
                        # Nutrition facts as a card
                        st.markdown("**🔍 Nutrition Facts**")
                        nutrition_html = f"""
                        <div style="background-color:#f0f2f6;border-radius:10px;padding:10px;margin:5px;">
                            <div style="font-size:1.2em;font-weight:bold;border-bottom:1px solid #ddd;padding-bottom:5px;margin-bottom:5px;">
                                {meal['nutrition']['calories']} calories
                            </div>
                            <div style="display:flex;justify-content:space-between;">
                                <div>
                                    <div style="font-weight:bold;">Protein</div>
                                    <div>{meal['nutrition']['protein']}g</div>
                                </div>
                                <div>
                                    <div style="font-weight:bold;">Carbs</div>
                                    <div>{meal['nutrition']['carbs']}g</div>
                                </div>
                                <div>
                                    <div style="font-weight:bold;">Fats</div>
                                    <div>{meal['nutrition']['fats']}g</div>
                                </div>
                            </div>
                            {f'<div style="margin-top:5px;"><b>Fiber:</b> {meal["nutrition"]["fiber"]}g</div>' if "fiber" in meal["nutrition"] else ''}
                        </div>
                        """
                        st.markdown(nutrition_html, unsafe_allow_html=True)

                    st.markdown("---")

        # Snacks section with cleaner design
        if 'snacks' in day_plan and day_plan['snacks']:
            st.markdown("#### Snacks")
            snack_cols = st.columns(min(3, len(day_plan['snacks'])))

            for i, (snack_time, snack) in enumerate(day_plan['snacks'].items()):
                with snack_cols[i % len(snack_cols)]:
                    st.markdown(f"**🍌 {snack_time.replace('_', ' ').title()}**")
                    st.markdown(f"*{snack['name']}*")
                    st.markdown(f"**Calories:** {snack['nutrition']['calories']} kcal")
                    st.markdown(
                        f"P: {snack['nutrition']['protein']}g | C: {snack['nutrition']['carbs']}g | F: {snack['nutrition']['fats']}g")

        # Hydration tracker
        if 'hydration' in day_plan:
            st.markdown("#### 💧 Hydration Tracker")

            hydration_col1, hydration_col2 = st.columns([1, 3])

            with hydration_col1:
                st.markdown(f"**Target:** {day_plan['hydration']['water']}")

            with hydration_col2:
                # Create water tracking cups (8 glasses = 2L typically)
                water_cols = st.columns(8)
                for i in range(8):
                    with water_cols[i]:
                        st.button("🥤", key=f"water_{selected_day}_{i}", help="Click to track water intake")

            if 'other' in day_plan['hydration'] and day_plan['hydration']['other']:
                st.markdown("**Additional recommended fluids:**")
                for item in day_plan['hydration']['other']:
                    st.markdown(f"- {item}")

    with tab2:
        # Nutrition profile with improved charts
        st.markdown("### Your Nutrition Profile")

        # Display BMI and other metrics
        profile_col1, profile_col2 = st.columns(2)

        with profile_col1:
            # Macronutrient chart
            st.markdown("#### Macronutrient Distribution")

            # Calculate percentages for the pie chart visualization
            protein_pct = plan['user_profile']['macros']['protein']['percent']
            carbs_pct = plan['user_profile']['macros']['carbs']['percent']
            fats_pct = plan['user_profile']['macros']['fats']['percent']

            # Create data for visualization
            macros_html = f"""
            <div style="text-align:center;">
                <div style="display:inline-block;width:200px;height:200px;border-radius:50%;background:conic-gradient(
                    #4CAF50 0% {protein_pct}%, 
                    #2196F3 {protein_pct}% {protein_pct + carbs_pct}%, 
                    #FFC107 {protein_pct + carbs_pct}% 100%);
                    position:relative;">
                </div>
                <div style="margin-top:20px;">
                    <span style="margin-right:15px;"><span style="color:#4CAF50;">■</span> Protein: {protein_pct}%</span>
                    <span style="margin-right:15px;"><span style="color:#2196F3;">■</span> Carbs: {carbs_pct}%</span>
                    <span><span style="color:#FFC107;">■</span> Fats: {fats_pct}%</span>
                </div>
            </div>
            """
            st.markdown(macros_html, unsafe_allow_html=True)

            # Display gram values
            st.markdown("#### Daily Macronutrient Targets")
            macro_cols = st.columns(3)
            macro_cols[0].metric("Protein", f"{plan['user_profile']['macros']['protein']['grams']}g")
            macro_cols[1].metric("Carbs", f"{plan['user_profile']['macros']['carbs']['grams']}g")
            macro_cols[2].metric("Fats", f"{plan['user_profile']['macros']['fats']['grams']}g")

        with profile_col2:
            # Micronutrients and considerations
            st.markdown("#### Key Micronutrients")
            if 'micro_nutrients' in plan['user_profile'] and plan['user_profile']['micro_nutrients']:
                for nutrient in plan['user_profile']['micro_nutrients']:
                    st.markdown(f"- **{nutrient}**")
            else:
                st.markdown("No specific micronutrients highlighted.")

            st.markdown("#### Health Considerations")
            if 'considerations' in plan['user_profile'] and plan['user_profile']['considerations']:
                for item in plan['user_profile']['considerations']:
                    st.markdown(f"- {item}")
            else:
                st.markdown("No specific health considerations noted.")

    with tab3:
        # Shopping list with interactive checklist
        st.markdown("### 🛒 Shopping List")

        # Group items by category
        for category, items in plan['shopping_list'].items():
            if items:
                st.markdown(f"#### {category.capitalize()}")
                for item in items:
                    st.checkbox(item, key=f"shop_{category}_{item}")

        # Print button for shopping list
        st.button("🖨️ Print Shopping List", help="Print your shopping list")

    with tab4:
        # Recommendations with icons
        if 'recommendations' in plan:
            st.markdown("### 💡 Recommendations & Tips")

            if 'general' in plan['recommendations']:
                st.markdown("#### 📝 General Advice")
                st.markdown(plan['recommendations']['general'])

            if 'supplements' in plan['recommendations'] and plan['recommendations']['supplements']:
                st.markdown("#### 💊 Recommended Supplements")
                for item in plan['recommendations']['supplements']:
                    st.markdown(f"- {item}")

            if 'tracking' in plan['recommendations']:
                st.markdown("#### 📊 Tracking Progress")
                st.markdown(plan['recommendations']['tracking'])

    # Actions section at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📥 Download Full Plan (JSON)",
            data=json.dumps(plan, indent=2),
            file_name=f"nutriai_plan_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

    with col2:
        # Share functionality with actionable options
        if st.button("📱 Share Plan", help="Share your nutrition plan"):
            # Create share links and display them when button is clicked
            st.session_state.show_share = True

        if st.session_state.get('show_share', False):
            share_options = st.columns(3)
            with share_options[0]:
                st.markdown(
                    "[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:?subject=My%20NutriAI%20Plan)")
            with share_options[1]:
                st.markdown(
                    "[![WhatsApp](https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)](https://wa.me/?text=Check%20out%20my%20nutrition%20plan)")
            with share_options[2]:
                st.markdown(
                    "[![Text](https://img.shields.io/badge/Text%20Message-4285F4?style=for-the-badge&logo=googlemessages&logoColor=white)](sms:?body=Check%20out%20my%20nutrition%20plan)")

    with col3:
        # Calendar functionality
        if st.button("📆 Add to Calendar", help="Add your meal plan to your calendar"):
            st.session_state.show_calendar = True

        if st.session_state.get('show_calendar', False):
            calendar_options = st.columns(3)
            with calendar_options[0]:
                st.markdown(
                    "[![Google](https://img.shields.io/badge/Google%20Calendar-4285F4?style=for-the-badge&logo=google-calendar&logoColor=white)](https://calendar.google.com/calendar/render?action=TEMPLATE&text=My%20Nutrition%20Plan)")
            with calendar_options[1]:
                st.markdown(
                    "[![Outlook](https://img.shields.io/badge/Outlook-0078D4?style=for-the-badge&logo=microsoft-outlook&logoColor=white)](https://outlook.office.com/calendar/action/compose)")
            with calendar_options[2]:
                st.markdown(
                    "[![Apple](https://img.shields.io/badge/Apple%20Calendar-FF2D20?style=for-the-badge&logo=apple&logoColor=white)](https://calendar.google.com/calendar/ical/)")


# ========== MAIN FUNCTION ==========
def main():
    st.set_page_config(page_title="NutriAI", page_icon="🍏", layout="wide")

    # Initialize session state variables
    if "nutrition_plan" not in st.session_state:
        st.session_state.nutrition_plan = None
    if "selected_day" not in st.session_state:
        st.session_state.selected_day = None
    if "water_count" not in st.session_state:
        st.session_state.water_count = 0
    if "show_share" not in st.session_state:
        st.session_state.show_share = False
    if "show_calendar" not in st.session_state:
        st.session_state.show_calendar = False

    user_data = get_user_input()

    if user_data and not st.session_state.nutrition_plan:
        with st.spinner("🔍 Analyzing your profile and creating your personalized plan..."):
            try:
                client = init_groq_client()
                prompt = create_prompt_template().format_prompt(**user_data)
                response = client.invoke(prompt.to_messages())

                # Extract and parse JSON with improved handling
                parsed = extract_and_parse_json(response.content)

                if parsed:
                    st.session_state.nutrition_plan = parsed
                else:
                    st.error("Failed to generate valid nutrition plan. Please try again.")
            except Exception as e:
                st.error(f"Error generating plan: {str(e)}")
                st.write("Please try again or check your Groq API key")

    if st.session_state.nutrition_plan:
        display_plan(st.session_state.nutrition_plan)

        if st.button("🔄 Generate New Plan"):
            st.session_state.nutrition_plan = None
            st.rerun()


if __name__ == "__main__":
    main()
