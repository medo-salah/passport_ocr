import sqlite3
import json
import datetime
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manage database operations for storing OCR results and analytics"""
    
    def __init__(self, db_path: str = "passport_ocr.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create OCR results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ocr_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        image_hash TEXT,
                        ocr_text TEXT,
                        mrz_text TEXT,
                        extracted_data TEXT,  -- JSON string
                        processing_time REAL,
                        language TEXT,
                        features_used TEXT,  -- JSON string
                        confidence_scores TEXT,  -- JSON string
                        error_message TEXT,
                        status TEXT DEFAULT 'success'
                    )
                """)
                
                # Create feedback table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        feedback_id TEXT UNIQUE NOT NULL,
                        session_id TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        image_hash TEXT,
                        feedback_type TEXT,
                        original_text TEXT,
                        corrected_text TEXT,
                        diff_ratio REAL,
                        additional_details TEXT,
                        status TEXT DEFAULT 'pending'
                    )
                """)
                
                # Create analytics events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        event_type TEXT,
                        details TEXT,  -- JSON string
                        user_agent TEXT,
                        ip_address TEXT
                    )
                """)
                
                # Create system logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        level TEXT,
                        message TEXT,
                        module TEXT,
                        session_id TEXT,
                        details TEXT  -- JSON string
                    )
                """)
                
                # Create user sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                        end_time DATETIME,
                        user_agent TEXT,
                        ip_address TEXT,
                        total_documents_processed INTEGER DEFAULT 0,
                        total_processing_time REAL DEFAULT 0,
                        status TEXT DEFAULT 'active'
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ocr_session ON ocr_results(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ocr_timestamp ON ocr_results(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_session ON analytics_events(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_type ON analytics_events(event_type)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def store_ocr_result(self, session_id: str, result_data: Dict[str, Any]) -> int:
        """Store OCR processing result in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO ocr_results (
                        session_id, image_hash, ocr_text, mrz_text, extracted_data,
                        processing_time, language, features_used, confidence_scores,
                        error_message, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    result_data.get('image_hash'),
                    result_data.get('ocr_text'),
                    result_data.get('mrz_text'),
                    json.dumps(result_data.get('extracted_data', {})),
                    result_data.get('processing_time'),
                    result_data.get('language'),
                    json.dumps(result_data.get('features_used', {})),
                    json.dumps(result_data.get('confidence_scores', [])),
                    result_data.get('error_message'),
                    result_data.get('status', 'success')
                ))
                
                result_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Stored OCR result with ID: {result_id}")
                return result_id
                
        except Exception as e:
            logger.error(f"Error storing OCR result: {e}")
            raise
    
    def store_feedback(self, feedback_data: Dict[str, Any]) -> int:
        """Store user feedback in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO feedback (
                        feedback_id, session_id, image_hash, feedback_type,
                        original_text, corrected_text, diff_ratio, additional_details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback_data.get('feedback_id'),
                    feedback_data.get('session_id'),
                    feedback_data.get('image_hash'),
                    feedback_data.get('feedback_type'),
                    feedback_data.get('original_text'),
                    feedback_data.get('corrected_text'),
                    feedback_data.get('diff_ratio'),
                    feedback_data.get('additional_details')
                ))
                
                feedback_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Stored feedback with ID: {feedback_id}")
                return feedback_id
                
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            raise
    
    def store_analytics_event(self, event_data: Dict[str, Any]) -> int:
        """Store analytics event in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO analytics_events (
                        session_id, event_type, details, user_agent, ip_address
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    event_data.get('session_id'),
                    event_data.get('event_type'),
                    json.dumps(event_data.get('details', {})),
                    event_data.get('user_agent'),
                    event_data.get('ip_address')
                ))
                
                event_id = cursor.lastrowid
                conn.commit()
                return event_id
                
        except Exception as e:
            logger.error(f"Error storing analytics event: {e}")
            raise
    
    def get_ocr_results(self, session_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Retrieve OCR results from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if session_id:
                    cursor.execute("""
                        SELECT * FROM ocr_results 
                        WHERE session_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (session_id, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM ocr_results 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                columns = [description[0] for description in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    # Parse JSON fields
                    if result['extracted_data']:
                        result['extracted_data'] = json.loads(result['extracted_data'])
                    if result['features_used']:
                        result['features_used'] = json.loads(result['features_used'])
                    if result['confidence_scores']:
                        result['confidence_scores'] = json.loads(result['confidence_scores'])
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving OCR results: {e}")
            return []
    
    def get_feedback(self, limit: int = 100) -> List[Dict]:
        """Retrieve feedback from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM feedback 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                columns = [description[0] for description in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving feedback: {e}")
            return []
    
    def get_analytics_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """Get analytics summary from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get date range
                start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).isoformat()
                
                # Total events
                cursor.execute("""
                    SELECT COUNT(*) FROM analytics_events 
                    WHERE timestamp >= ?
                """, (start_date,))
                total_events = cursor.fetchone()[0]
                
                # Unique sessions
                cursor.execute("""
                    SELECT COUNT(DISTINCT session_id) FROM analytics_events 
                    WHERE timestamp >= ?
                """, (start_date,))
                unique_sessions = cursor.fetchone()[0]
                
                # Events by type
                cursor.execute("""
                    SELECT event_type, COUNT(*) FROM analytics_events 
                    WHERE timestamp >= ?
                    GROUP BY event_type
                """, (start_date,))
                events_by_type = dict(cursor.fetchall())
                
                # OCR results count
                cursor.execute("""
                    SELECT COUNT(*) FROM ocr_results 
                    WHERE timestamp >= ?
                """, (start_date,))
                ocr_results_count = cursor.fetchone()[0]
                
                # Average processing time
                cursor.execute("""
                    SELECT AVG(processing_time) FROM ocr_results 
                    WHERE timestamp >= ? AND processing_time IS NOT NULL
                """, (start_date,))
                avg_processing_time = cursor.fetchone()[0] or 0
                
                return {
                    "total_events": total_events,
                    "unique_sessions": unique_sessions,
                    "events_by_type": events_by_type,
                    "ocr_results_count": ocr_results_count,
                    "avg_processing_time": avg_processing_time,
                    "period_days": days_back
                }
                
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days_to_keep)).isoformat()
                
                # Clean up old analytics events
                cursor.execute("DELETE FROM analytics_events WHERE timestamp < ?", (cutoff_date,))
                analytics_deleted = cursor.rowcount
                
                # Clean up old OCR results (keep more recent ones)
                cursor.execute("DELETE FROM ocr_results WHERE timestamp < ?", (cutoff_date,))
                ocr_deleted = cursor.rowcount
                
                # Clean up old system logs
                cursor.execute("DELETE FROM system_logs WHERE timestamp < ?", (cutoff_date,))
                logs_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up old data: {analytics_deleted} analytics events, {ocr_deleted} OCR results, {logs_deleted} log entries")
                
                return {
                    "analytics_deleted": analytics_deleted,
                    "ocr_deleted": ocr_deleted,
                    "logs_deleted": logs_deleted
                }
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {}
    
    def export_data(self, table_name: str, format: str = "json") -> str:
        """Export data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"SELECT * FROM {table_name}")
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                if format == "json":
                    data = []
                    for row in rows:
                        data.append(dict(zip(columns, row)))
                    return json.dumps(data, indent=2, default=str)
                
                elif format == "csv":
                    import csv
                    import io
                    
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(columns)
                    writer.writerows(rows)
                    return output.getvalue()
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return ""
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Table row counts
                tables = ['ocr_results', 'feedback', 'analytics_events', 'system_logs', 'user_sessions']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # Database file size
                if os.path.exists(self.db_path):
                    stats["db_size_mb"] = os.path.getsize(self.db_path) / (1024 * 1024)
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

# Global database manager instance
db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get or create database manager instance"""
    global db_manager
    if db_manager is None:
        db_path = st.session_state.get("admin_db_path", "passport_ocr.db")
        db_manager = DatabaseManager(db_path)
    return db_manager

def is_database_enabled() -> bool:
    """Check if database storage is enabled"""
    return st.session_state.get("admin_enable_database", False)
