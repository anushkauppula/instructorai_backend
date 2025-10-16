#!/usr/bin/env python3
"""
Supabase Setup Script for Majors Exploration AI Assistant
This script helps set up the required Supabase configuration.
"""

import os
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_supabase():
    """Set up Supabase tables and storage bucket."""
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        return False
    
    try:
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase")
        
        # Check if academic_explorations table exists
        try:
            result = supabase.table("academic_explorations").select("id").limit(1).execute()
            print("✅ academic_explorations table already exists")
        except Exception as e:
            print(f"⚠️  academic_explorations table doesn't exist: {e}")
            print("📝 Please run the SQL commands in setup_supabase.sql in your Supabase Dashboard")
        
        # Check if audio-files bucket exists
        try:
            buckets = supabase.storage.list_buckets()
            bucket_names = [bucket.name for bucket in buckets]
            
            if "audio-files" in bucket_names:
                print("✅ audio-files storage bucket exists")
            else:
                print("⚠️  audio-files storage bucket doesn't exist")
                print("📝 Please create the 'audio-files' bucket in Supabase Storage")
                
        except Exception as e:
            print(f"⚠️  Could not check storage buckets: {e}")
            print("📝 Please verify your Supabase Storage configuration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        return False

def main():
    """Main setup function."""
    print("🎓 Setting up Supabase for Majors Exploration AI Assistant")
    print("=" * 60)
    
    if setup_supabase():
        print("\n✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run the SQL commands in setup_supabase.sql in your Supabase Dashboard")
        print("2. Create the 'audio-files' storage bucket in Supabase Storage")
        print("3. Test your application")
    else:
        print("\n❌ Setup failed. Please check your configuration.")

if __name__ == "__main__":
    main()
