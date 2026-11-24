from app.services.mysql_service import db_service
from sqlalchemy import text

def migrate():
    with db_service.get_session() as s:
        cols = [c[0] for c in s.execute(text("DESCRIBE functions")).fetchall()]
        print("Existing columns:", cols)
        
        if "verification_status" not in cols:
            s.execute(text("ALTER TABLE functions ADD COLUMN verification_status VARCHAR(50) DEFAULT 'verified'"))
            print("Added verification_status")
        if "conflict_status" not in cols:
            s.execute(text("ALTER TABLE functions ADD COLUMN conflict_status VARCHAR(50) DEFAULT 'clean'"))
            print("Added conflict_status")
        if "parent_function_id" not in cols:
            s.execute(text("ALTER TABLE functions ADD COLUMN parent_function_id INT NULL"))
            print("Added parent_function_id")
            
        s.commit()
        print("Schema migration completed successfully!")

if __name__ == "__main__":
    migrate()
