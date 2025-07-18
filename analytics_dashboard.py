import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import datetime
from collections import defaultdict, Counter
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class AnalyticsTracker:
    """Track usage patterns and error rates for the passport OCR application"""
    
    def __init__(self, analytics_dir: str = "analytics_data"):
        self.analytics_dir = analytics_dir
        os.makedirs(analytics_dir, exist_ok=True)
        
    def log_event(self, event_type: str, details: Dict[str, Any] = None):
        """Log an analytics event"""
        try:
            timestamp = datetime.datetime.now()
            event = {
                "timestamp": timestamp.isoformat(),
                "date": timestamp.strftime("%Y-%m-%d"),
                "hour": timestamp.hour,
                "event_type": event_type,
                "details": details or {},
                "session_id": getattr(st.session_state, 'session_id', 'unknown')
            }
            
            # Save to daily log file
            log_file = os.path.join(self.analytics_dir, f"events_{timestamp.strftime('%Y-%m-%d')}.jsonl")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
                
        except Exception as e:
            logger.error(f"Error logging analytics event: {e}")
    
    def load_events(self, days_back: int = 30) -> List[Dict]:
        """Load events from the last N days"""
        events = []
        end_date = datetime.datetime.now()
        
        for i in range(days_back):
            date = end_date - datetime.timedelta(days=i)
            log_file = os.path.join(self.analytics_dir, f"events_{date.strftime('%Y-%m-%d')}.jsonl")
            
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                events.append(json.loads(line))
                except Exception as e:
                    logger.error(f"Error loading events from {log_file}: {e}")
        
        return events
    
    def get_usage_stats(self, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        events = self.load_events(days_back)
        
        if not events:
            return {"total_events": 0, "message": "No data available"}
        
        df = pd.DataFrame(events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = pd.to_datetime(df['date'])
        
        stats = {
            "total_events": len(events),
            "unique_sessions": df['session_id'].nunique(),
            "date_range": {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat()
            },
            "events_by_type": df['event_type'].value_counts().to_dict(),
            "daily_usage": df.groupby('date').size().to_dict(),
            "hourly_usage": df.groupby('hour').size().to_dict(),
            "error_rate": self._calculate_error_rate(df),
            "processing_times": self._extract_processing_times(df),
            "feature_usage": self._analyze_feature_usage(df),
            "language_usage": self._analyze_language_usage(df),
            "device_types": self._analyze_device_types(df)
        }
        
        return stats
    
    def _calculate_error_rate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate error rates by type"""
        total_processing = len(df[df['event_type'].isin(['ocr_success', 'ocr_error', 'mrz_success', 'mrz_error'])])

        if total_processing == 0:
            return {
                "overall": 0.0,
                "ocr_error_rate": 0.0,
                "mrz_error_rate": 0.0
            }

        error_events = df[df['event_type'].str.contains('error', na=False)]

        # Calculate OCR error rate
        ocr_total = len(df[df['event_type'].isin(['ocr_success', 'ocr_error'])])
        ocr_errors = len(df[df['event_type'] == 'ocr_error'])
        ocr_error_rate = (ocr_errors / max(1, ocr_total)) * 100 if ocr_total > 0 else 0.0

        # Calculate MRZ error rate
        mrz_total = len(df[df['event_type'].isin(['mrz_success', 'mrz_error'])])
        mrz_errors = len(df[df['event_type'] == 'mrz_error'])
        mrz_error_rate = (mrz_errors / max(1, mrz_total)) * 100 if mrz_total > 0 else 0.0

        return {
            "overall": len(error_events) / total_processing * 100,
            "ocr_error_rate": ocr_error_rate,
            "mrz_error_rate": mrz_error_rate
        }
    
    def _extract_processing_times(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract processing time statistics"""
        processing_events = df[df['event_type'] == 'processing_complete']
        
        if processing_events.empty:
            return {"avg": 0.0, "min": 0.0, "max": 0.0}
        
        times = []
        for _, event in processing_events.iterrows():
            if 'processing_time' in event.get('details', {}):
                times.append(event['details']['processing_time'])
        
        if not times:
            return {"avg": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "count": len(times)
        }
    
    def _analyze_feature_usage(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze which features are used most"""
        feature_usage = defaultdict(int)
        
        for _, event in df.iterrows():
            details = event.get('details', {})
            
            # Track specific features
            if event['event_type'] == 'feature_used':
                feature = details.get('feature', 'unknown')
                feature_usage[feature] += 1
            
            # Track preprocessing options
            if 'preprocessing' in details:
                for option, enabled in details['preprocessing'].items():
                    if enabled:
                        feature_usage[f"preprocessing_{option}"] += 1
        
        return dict(feature_usage)
    
    def _analyze_language_usage(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze language usage patterns"""
        language_usage = defaultdict(int)
        
        for _, event in df.iterrows():
            details = event.get('details', {})
            if 'language' in details:
                language_usage[details['language']] += 1
        
        return dict(language_usage)
    
    def _analyze_device_types(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze device type usage"""
        device_usage = defaultdict(int)
        
        for _, event in df.iterrows():
            details = event.get('details', {})
            if 'device_type' in details:
                device_usage[details['device_type']] += 1
        
        return dict(device_usage)

def display_analytics_dashboard():
    """Display the analytics dashboard in Streamlit"""
    st.header("📊 Analytics Dashboard")
    st.markdown("Track usage patterns and error rates to guide improvements")
    
    # Initialize analytics tracker
    tracker = AnalyticsTracker()
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox(
            "Analysis Period",
            [7, 14, 30, 60, 90],
            index=2,
            help="Number of days to analyze"
        )
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Load and display statistics
    with st.spinner("Loading analytics data..."):
        stats = tracker.get_usage_stats(days_back)
    
    if stats.get("total_events", 0) == 0:
        st.info("No analytics data available yet. Start using the application to see statistics.")
        return
    
    # Overview metrics
    st.subheader("📈 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", stats["total_events"])
    
    with col2:
        st.metric("Unique Sessions", stats["unique_sessions"])
    
    with col3:
        error_rate = stats.get("error_rate", {}).get("overall", 0.0)
        st.metric("Error Rate", f"{error_rate:.1f}%")

    with col4:
        avg_processing_time = stats.get("processing_times", {}).get("avg", 0.0)
        st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
    
    # Usage patterns
    st.subheader("📅 Usage Patterns")
    
    # Daily usage chart
    daily_usage = stats.get("daily_usage", {})
    if daily_usage:
        daily_df = pd.DataFrame(list(daily_usage.items()), columns=["Date", "Events"])
        daily_df["Date"] = pd.to_datetime(daily_df["Date"])

        fig_daily = px.line(daily_df, x="Date", y="Events", title="Daily Usage")
        st.plotly_chart(fig_daily, use_container_width=True)
    else:
        st.info("No daily usage data available yet.")

    # Hourly usage chart
    hourly_usage = stats.get("hourly_usage", {})
    if hourly_usage:
        hourly_df = pd.DataFrame(list(hourly_usage.items()), columns=["Hour", "Events"])

        fig_hourly = px.bar(hourly_df, x="Hour", y="Events", title="Usage by Hour of Day")
        st.plotly_chart(fig_hourly, use_container_width=True)
    else:
        st.info("No hourly usage data available yet.")
    
    # Error analysis
    st.subheader("⚠️ Error Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Error rates by type
        error_rate_data = stats.get("error_rate", {})
        error_data = {
            "OCR Errors": error_rate_data.get("ocr_error_rate", 0.0),
            "MRZ Errors": error_rate_data.get("mrz_error_rate", 0.0)
        }

        fig_errors = px.bar(
            x=list(error_data.keys()),
            y=list(error_data.values()),
            title="Error Rates by Type (%)"
        )
        st.plotly_chart(fig_errors, use_container_width=True)
    
    with col2:
        # Event types distribution
        events_by_type = stats.get("events_by_type", {})
        if events_by_type:
            fig_events = px.pie(
                values=list(events_by_type.values()),
                names=list(events_by_type.keys()),
                title="Event Types Distribution"
            )
            st.plotly_chart(fig_events, use_container_width=True)
        else:
            st.info("No event type data available yet.")

    # Feature usage
    st.subheader("🔧 Feature Usage")

    feature_usage = stats.get("feature_usage", {})
    if feature_usage:
        feature_df = pd.DataFrame(list(feature_usage.items()), columns=["Feature", "Usage Count"])
        feature_df = feature_df.sort_values("Usage Count", ascending=True)

        fig_features = px.bar(feature_df, x="Usage Count", y="Feature", orientation="h", title="Feature Usage")
        st.plotly_chart(fig_features, use_container_width=True)
    else:
        st.info("No feature usage data available yet.")

    # Language usage
    language_usage = stats.get("language_usage", {})
    if language_usage:
        st.subheader("🌍 Language Usage")
        lang_df = pd.DataFrame(list(language_usage.items()), columns=["Language", "Count"])

        fig_lang = px.pie(lang_df, values="Count", names="Language", title="Language Distribution")
        st.plotly_chart(fig_lang, use_container_width=True)
    else:
        st.info("No language usage data available yet.")
    
    # Performance metrics
    st.subheader("⚡ Performance Metrics")

    processing_times = stats.get("processing_times", {})
    if processing_times.get("count", 0) > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Min Processing Time", f"{processing_times.get('min', 0.0):.2f}s")

        with col2:
            st.metric("Max Processing Time", f"{processing_times.get('max', 0.0):.2f}s")

        with col3:
            st.metric("Processed Documents", processing_times.get("count", 0))
    else:
        st.info("No processing time data available yet.")
    
    # Raw data export
    st.subheader("📤 Data Export")
    
    if st.button("Export Analytics Data"):
        # Create export data
        export_data = {
            "generated_at": datetime.datetime.now().isoformat(),
            "period_days": days_back,
            "statistics": stats
        }
        
        # Convert to JSON
        json_data = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            label="Download Analytics Report",
            data=json_data,
            file_name=f"analytics_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Global analytics tracker instance
analytics_tracker = AnalyticsTracker()

def log_analytics_event(event_type: str, details: Dict[str, Any] = None):
    """Convenience function to log analytics events"""
    try:
        analytics_tracker.log_event(event_type, details)

        # Also store in database if enabled
        try:
            import streamlit as st
            from database_manager import is_database_enabled, get_database_manager

            if is_database_enabled() and st.session_state.get("admin_store_analytics", True):
                db_manager = get_database_manager()

                event_data = {
                    "session_id": getattr(st.session_state, 'session_id', 'unknown'),
                    "event_type": event_type,
                    "details": details or {},
                    "user_agent": "unknown",  # Could be enhanced to get real user agent
                    "ip_address": "unknown"   # Could be enhanced to get real IP
                }

                db_manager.store_analytics_event(event_data)

        except Exception as db_error:
            logger.error(f"Failed to store analytics event in database: {db_error}")

    except Exception as e:
        logger.error(f"Failed to log analytics event: {e}")
