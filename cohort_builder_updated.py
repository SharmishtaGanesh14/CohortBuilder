#!/usr/bin/env python3
"""
Dynamic Cohort Builder

A medical cohort analysis tool that converts natural language queries
into structured filters and visualizes patient data dynamically.

Features:
- Natural language to filter conversion
- Live confidence scoring
- Dynamic chart generation
- Universal dataset compatibility
"""

import os
import re
import json
import difflib
from typing import Dict, List, Any

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# OpenAI integration with graceful fallback
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class CohortBuilder:
    """Main application class for building medical cohorts."""
    
    def __init__(self):
        self.initialize_data()
        self.setup_session_state()
    
    def initialize_data(self):
        """Set up empty dataset - requires user upload."""
        self.merged_df = pd.DataFrame()
        print("📋 Ready for user data upload")
    
    def setup_session_state(self):
        """Initialize Streamlit session state variables."""
        session_vars = {
            'current_filters': [],
            'clarification_needed': None,
            'llm_confidence': 0.0,
            'interaction_count': 0,
            'uploaded_data': None
        }
        
        for var, default in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default
    
    def parse_query_with_llm(self, query: str, clarification: str = None) -> Dict[str, Any]:
        """Parse natural language query using OpenAI LLM."""
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or not OPENAI_AVAILABLE:
            return self._mock_parse_response(query)
        
        try:
            # Build dynamic prompt based on available columns
            available_cols = list(st.session_state.uploaded_data.columns) if st.session_state.uploaded_data is not None else []
            cols_str = ", ".join(available_cols[:10])
            if len(available_cols) > 10:
                cols_str += f" and {len(available_cols) - 10} more columns"
            
            prompt = f"""
Parse this medical cohort query into structured filters.
Return JSON with: filters[], confidence (0-1), clarification_needed (string or null)

Available columns: {cols_str}

Query: "{query}"
"""
            
            if clarification:
                prompt += f"\nUser clarification: {clarification}"
            
            # Call OpenAI API
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate and normalize response
            result.setdefault('filters', [])
            result.setdefault('confidence', 0.5)
            result.setdefault('clarification_needed', None)
            
            return result
            
        except Exception as e:
            print(f"LLM parse failed: {e}")
            return self._mock_parse_response(query)
    
    def _mock_parse_response(self, query: str) -> Dict[str, Any]:
        """Parse natural language query using pattern matching.
        
        Falls back to regex-based parsing when LLM is unavailable.
        Dynamically adapts to available dataset columns.
        """
        query_lower = query.lower()
        filters = []
        
        # Get available columns from uploaded data
        available_cols = list(st.session_state.uploaded_data.columns) if st.session_state.uploaded_data is not None else []
        
        # Find relevant columns
        age_col = self._find_column_by_keywords(available_cols, ['age', 'year', 'old'])
        gender_col = self._find_column_by_keywords(available_cols, ['gender', 'sex'])
        
        # Parse age-related queries
        if age_col:
            filters.extend(self._parse_age_filters(query_lower, age_col))
        
        # Parse gender-related queries
        if gender_col:
            filters.extend(self._parse_gender_filters(query_lower, gender_col))
        
        return {
            'filters': filters, 
            'confidence': 0.7, 
            'clarification_needed': None, 
            'explanation': 'Pattern-based parsing'
        }
    
    def _find_column_by_keywords(self, columns: List[str], keywords: List[str]) -> str:
        """Find first column matching any of the given keywords."""
        for col in columns:
            if any(keyword in col.lower() for keyword in keywords):
                return col
        return None
    
    def _parse_age_filters(self, query: str, age_col: str) -> List[Dict]:
        """Parse age-related filter patterns."""
        filters = []
        
        # Handle explicit age comparisons (age > 65)
        age_match = re.search(r'age\s*[><=]\s*(\d+)', query)
        if age_match:
            age_val = int(age_match.group(1))
            operator_match = re.search(r'age\s*([><=])\s*\d+', query)
            operator = operator_match.group(1) if operator_match else '>'
            filters.append({
                'column': age_col, 
                'operator': operator, 
                'value': age_val, 
                'type': 'inclusion'
            })
        
        # Handle "elderly" - use 75th percentile of actual data
        elif 'elderly' in query or 'age' in query:
            if st.session_state.uploaded_data is not None and not st.session_state.uploaded_data[age_col].empty:
                try:
                    elderly_threshold = int(st.session_state.uploaded_data[age_col].quantile(0.75))
                except:
                    elderly_threshold = 60  # Conservative fallback
                
                filters.append({
                    'column': age_col, 
                    'operator': '>', 
                    'value': elderly_threshold, 
                    'type': 'inclusion'
                })
        
        # Handle "under X" and "over X" patterns
        under_match = re.search(r'under\s+(\d+)', query)
        if under_match:
            filters.append({
                'column': age_col, 
                'operator': '<', 
                'value': int(under_match.group(1)), 
                'type': 'inclusion'
            })
        
        over_match = re.search(r'over\s+(\d+)', query)
        if over_match:
            filters.append({
                'column': age_col, 
                'operator': '>', 
                'value': int(over_match.group(1)), 
                'type': 'inclusion'
            })
        
        return filters
    
    def _parse_gender_filters(self, query: str, gender_col: str) -> List[Dict]:
        """Parse gender-related filter patterns."""
        filters = []
        
        # Get actual gender values from dataset
        if st.session_state.uploaded_data is not None:
            gender_values = [str(val).strip().lower() for val in st.session_state.uploaded_data[gender_col].unique() if pd.notna(val)]
            male_values = [val for val in gender_values if any(word in val for word in ['male', 'm'])]
            female_values = [val for val in gender_values if any(word in val for word in ['female', 'f'])]
            
            # Parse gender patterns
            has_male = ' male' in query or query.startswith('male') or query.endswith('male')
            has_female = ' female' in query or query.startswith('female') or query.endswith('female')
            
            if 'exclude male' in query and male_values:
                filters.append({
                    'column': gender_col, 
                    'operator': '!=', 
                    'value': male_values[0].title(), 
                    'type': 'exclusion'
                })
            elif 'exclude female' in query and female_values:
                filters.append({
                    'column': gender_col, 
                    'operator': '!=', 
                    'value': female_values[0].title(), 
                    'type': 'exclusion'
                })
            elif has_male and not has_female and male_values:
                filters.append({
                    'column': gender_col, 
                    'operator': '=', 
                    'value': male_values[0].title(), 
                    'type': 'inclusion'
                })
            elif has_female and not has_male and female_values:
                filters.append({
                    'column': gender_col, 
                    'operator': '=', 
                    'value': female_values[0].title(), 
                    'type': 'inclusion'
                })
        
        return filters
    
    def apply_filters(self, df: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
        """Apply structured filters to dataframe."""
        filtered_df = df.copy()
        
        for filter_item in filters:
            column = filter_item['column']
            operator = filter_item['operator']
            value = filter_item['value']
            filter_type = filter_item['type']
            
            if column not in filtered_df.columns:
                continue
                
            if filter_type == 'exclusion':
                if operator == '!=':
                    filtered_df = filtered_df[filtered_df[column] != value]
            else:  # inclusion
                if operator == '=':
                    filtered_df = filtered_df[filtered_df[column] == value]
                elif operator == '!=':
                    filtered_df = filtered_df[filtered_df[column] != value]
                elif operator == '>':
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif operator == '<':
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif operator == '>=':
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif operator == '<=':
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif operator == 'isin':
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                elif operator == 'notin':
                    filtered_df = filtered_df[~filtered_df[column].isin(value)]
                elif operator == 'isna':
                    filtered_df = filtered_df[filtered_df[column].isna()]
                elif operator == 'notna':
                    filtered_df = filtered_df[filtered_df[column].notna()]
        
        return filtered_df
    
    def create_charts(self, cohort_df: pd.DataFrame):
        """Create completely dynamic charts based on dataset columns."""
        if cohort_df.empty:
            return
        
        # Find columns dynamically
        numeric_cols = cohort_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = cohort_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID columns from analysis
        id_keywords = ['id', 'identifier', 'subject', 'patient']
        numeric_cols = [col for col in numeric_cols if not any(keyword in col.lower() for keyword in id_keywords)]
        categorical_cols = [col for col in categorical_cols if not any(keyword in col.lower() for keyword in id_keywords)]
        
        # Determine what charts to show based on available data
        chart_types = []
        
        # Age/numeric histogram
        age_col = None
        for col in numeric_cols:
            if any(age_word in col.lower() for age_word in ['age', 'year', 'old']):
                age_col = col
                chart_types.append(('histogram', col, 'Age Distribution'))
                break
        
        # Gender pie chart
        gender_col = None
        for col in categorical_cols:
            if any(gender_word in col.lower() for gender_word in ['gender', 'sex']):
                gender_col = col
                chart_types.append(('pie', col, 'Gender Distribution'))
                break
        
        # Categorical bar chart
        cat_col = None
        for col in categorical_cols:
            if col != gender_col and len(cohort_df[col].unique()) <= 10:
                cat_col = col
                chart_types.append(('bar', col, f'{col.title()} Distribution'))
                break
        
        # Status/diagnosis chart
        status_col = None
        for col in categorical_cols:
            if any(status_word in col.lower() for status_word in ['status', 'diagnosis', 'dx', 'condition', 'malignancy', 'relapse']):
                if col != gender_col and col != cat_col:
                    status_col = col
                    chart_types.append(('bar', col, f'{col.title()} Breakdown'))
                    break
        
        # Limit to 4 charts max
        chart_types = chart_types[:4]
        
        if not chart_types:
            st.info("📊 No suitable columns found for visualization")
            return
        
        # Create dynamic subplot specs
        rows, cols = 2, 2
        specs = [[{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]]
        
        # Update specs for pie charts
        for i, (chart_type, col, title) in enumerate(chart_types):
            if chart_type == 'pie':
                row = i // 2
                col_pos = i % 2
                specs[row][col_pos] = {"type": "domain"}
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[chart[2] for chart in chart_types] + ['', ''][:4-len(chart_types)],
            specs=specs
        )
        
        # Add traces based on chart types
        for i, (chart_type, col, title) in enumerate(chart_types):
            row = (i // 2) + 1
            col_pos = (i % 2) + 1
            
            if chart_type == 'histogram':
                fig.add_trace(go.Histogram(x=cohort_df[col], name=col), row=row, col=col_pos)
            elif chart_type == 'pie':
                value_counts = cohort_df[col].value_counts()
                fig.add_trace(go.Pie(labels=value_counts.index, values=value_counts.values, name=col), row=row, col=col_pos)
            elif chart_type == 'bar':
                value_counts = cohort_df[col].value_counts().head(8)
                fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values, name=col), row=row, col=col_pos)
        
        fig.update_layout(height=500, showlegend=False, title_text=f"📊 Dataset Overview ({len(cohort_df)} records)")
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main application entry point."""
        st.set_page_config(page_title="Dynamic Cohort Builder", layout="wide")
        st.title("🏥 Dynamic Cohort Builder")
        
        # Sidebar for filters
        with st.sidebar:
            st.header("🔍 Build Cohort")
            
            # Natural language input
            query = st.text_input("Natural Language Query:", placeholder="e.g., 'elderly female patients'")
            
            # Live confidence update when typing - use local variable to avoid rerun loop
            live_confidence = 0.0
            if query and st.session_state.uploaded_data is not None:
                try:
                    mock_parsed = self._mock_parse_response(query)
                    live_confidence = mock_parsed['confidence']
                except:
                    live_confidence = 0.0
            
            # Handle clarification
            if st.session_state.clarification_needed:
                st.warning(st.session_state.clarification_needed)
                clarification = st.text_input("Clarify:", key="clarify")
                if st.button("Submit Clarification"):
                    parsed = self.parse_query_with_llm(query, clarification)
                    st.session_state.current_filters = parsed['filters']
                    st.session_state.llm_confidence = parsed['confidence']
                    st.session_state.clarification_needed = parsed['clarification_needed']
                    st.session_state.interaction_count += 1
                    st.rerun()
            
            # Main parse button
            if st.button("Parse Query", type="primary"):
                if query:
                    parsed = self.parse_query_with_llm(query)
                    st.session_state.current_filters = parsed['filters']
                    st.session_state.llm_confidence = parsed['confidence']
                    st.session_state.clarification_needed = parsed['clarification_needed']
                    st.session_state.interaction_count = 0
                    
                    if parsed['clarification_needed']:
                        st.rerun()
            
            # Display active filters
            if st.session_state.current_filters:
                st.write("🎯 Active Filters:")
                for i, f in enumerate(st.session_state.current_filters):
                    emoji = "✅" if f['type'] == 'inclusion' else "❌"
                    chip_text = f"{emoji} {f['column']} {f['operator']} {f['value']}"
                    if st.button(chip_text, key=f"chip_{i}"):
                        st.session_state.current_filters.pop(i)
                        st.rerun()
            
            # Confidence bar
            confidence_to_show = live_confidence if live_confidence > 0 else st.session_state.llm_confidence
            if confidence_to_show > 0:
                st.write("📊 Confidence (Live Updates as You Type):")
                st.progress(confidence_to_show)
                st.write(f"{confidence_to_show:.1%}")
                st.caption("💡 Confidence updates in real-time. Click 'Parse Query' for full LLM analysis.")
            
            # Clear button
            if st.button("🗑️ Clear All"):
                st.session_state.current_filters = []
                st.session_state.llm_confidence = 0.0
                st.session_state.clarification_needed = None
                st.rerun()
            
            st.divider()
            self.handle_upload()
        
        # Main content area
        data_to_use = st.session_state.uploaded_data if st.session_state.uploaded_data is not None else self.merged_df
        
        if data_to_use.empty:
            st.info("👆 Please upload a CSV file to begin building your cohort.")
            return
        
        # Apply filters and show results
        current_cohort = self.apply_filters(data_to_use, st.session_state.current_filters)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📋 Cohort Data ({len(current_cohort)} records)")
            st.dataframe(current_cohort, use_container_width=True)
        
        with col2:
            st.subheader("📊 Statistics")
            if not current_cohort.empty:
                for col in current_cohort.columns:
                    if current_cohort[col].dtype in ['int64', 'float64']:
                        st.metric(f"{col.title()} Mean", round(current_cohort[col].mean(), 2))
                    elif current_cohort[col].dtype == 'object':
                        unique_count = current_cohort[col].nunique()
                        st.metric(f"{col.title()} Unique", unique_count)
        
        # Charts section
        if not current_cohort.empty:
            st.divider()
            st.subheader("📈 Visualizations")
            self.create_charts(current_cohort)
    
    def handle_upload(self):
        """Handle custom CSV upload with flexible ID detection."""
        uploaded_file = st.file_uploader(
            "📁 Upload Custom CSV (We'll detect the patient ID column)", 
            type=["csv"],
            key="csv_uploader"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Auto-detect ID column
                id_col = self._detect_id_column(df)
                
                if id_col:
                    st.success(f"✅ Detected ID column: {id_col}")
                    st.session_state.uploaded_data = df
                    st.session_state.current_filters = []
                    st.session_state.llm_confidence = 0.0
                else:
                    st.warning("⚠️ Couldn't auto-detect patient ID column. Please select it manually:")
                    id_col = st.selectbox("Select ID column:", df.columns.tolist())
                    if st.button("Use Selected Column"):
                        st.session_state.uploaded_data = df
                        st.session_state.current_filters = []
                        st.session_state.llm_confidence = 0.0
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")
    
    def _detect_id_column(self, df: pd.DataFrame) -> str:
        """Automatically detect the patient ID column."""
        id_keywords = ['patient_id', 'subject_id', 'case_id', 'submitter_id', 'participant_id', 'id']
        
        for col in df.columns:
            col_clean = col.lower().strip().replace(' ', '_')
            if any(keyword in col_clean for keyword in id_keywords):
                return col
        
        return None


def main():
    """Application entry point."""
    app = CohortBuilder()
    app.run()


if __name__ == "__main__":
    main()
