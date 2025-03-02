"""
Alerts Module

This script handles alerts and notifications for the trading system.
"""

import os
import json
import logging
import smtplib
import telepot
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/alerts.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Class to manage alerts and notifications
    """
    
    def __init__(self, config_path='../config/alerts_config.json'):
        """
        Initialize the alert manager
        
        Args:
            config_path (str): Path to the alerts configuration file
        """
        self.config = {
            'telegram': {
                'enabled': False,
                'token': None,
                'chat_id': None
            },
            'slack': {
                'enabled': False,
                'webhook_url': None
            },
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': None,
                'password': None,
                'from_email': None,
                'to_email': None
            }
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs("../logs", exist_ok=True)
        
        # Load configuration if available
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Update config with loaded values
                    for key in self.config:
                        if key in loaded_config:
                            self.config[key].update(loaded_config[key])
                logger.info("Loaded alerts configuration from file")
            except Exception as e:
                logger.error(f"Error loading alerts configuration: {e}")
        
        # Initialize Telegram bot if enabled
        self.telegram_bot = None
        if self.config['telegram']['enabled'] and self.config['telegram']['token']:
            try:
                self.telegram_bot = telepot.Bot(self.config['telegram']['token'])
                logger.info("Telegram bot initialized")
            except Exception as e:
                logger.error(f"Error initializing Telegram bot: {e}")
    
    def save_config(self, config_path='../config/alerts_config.json'):
        """
        Save alerts configuration to file
        
        Args:
            config_path (str): Path to save the configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def send_telegram_alert(self, message):
        """
        Send an alert via Telegram
        
        Args:
            message (str): The message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.config['telegram']['enabled']:
            logger.info("Telegram alerts are disabled")
            return False
        
        if not self.telegram_bot or not self.config['telegram']['chat_id']:
            logger.error("Telegram bot or chat ID not available")
            return False
        
        try:
            self.telegram_bot.sendMessage(self.config['telegram']['chat_id'], message)
            logger.info(f"Telegram alert sent: {message}")
            return True
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
            return False
    
    def send_slack_alert(self, message):
        """
        Send an alert via Slack
        
        Args:
            message (str): The message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.config['slack']['enabled']:
            logger.info("Slack alerts are disabled")
            return False
        
        if not self.config['slack']['webhook_url']:
            logger.error("Slack webhook URL not available")
            return False
        
        try:
            payload = {
                'text': message
            }
            response = requests.post(
                self.config['slack']['webhook_url'],
                json=payload
            )
            if response.status_code == 200:
                logger.info(f"Slack alert sent: {message}")
                return True
            else:
                logger.error(f"Error sending Slack alert: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False
    
    def send_email_alert(self, subject, message):
        """
        Send an alert via email
        
        Args:
            subject (str): The email subject
            message (str): The email message
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.config['email']['enabled']:
            logger.info("Email alerts are disabled")
            return False
        
        email_config = self.config['email']
        if not all([
            email_config['smtp_server'],
            email_config['smtp_port'],
            email_config['username'],
            email_config['password'],
            email_config['from_email'],
            email_config['to_email']
        ]):
            logger.error("Email configuration incomplete")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']
            msg['Subject'] = subject
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False
    
    def send_alert(self, message, subject=None, channels=None):
        """
        Send an alert to all enabled channels
        
        Args:
            message (str): The message to send
            subject (str): The email subject (if None, a default subject is used)
            channels (list): List of channels to send to (if None, all enabled channels are used)
            
        Returns:
            dict: Dictionary with results for each channel
        """
        if subject is None:
            subject = f"Trading Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        if channels is None:
            channels = ['telegram', 'slack', 'email']
        
        results = {}
        
        if 'telegram' in channels:
            results['telegram'] = self.send_telegram_alert(message)
        
        if 'slack' in channels:
            results['slack'] = self.send_slack_alert(message)
        
        if 'email' in channels:
            results['email'] = self.send_email_alert(subject, message)
        
        return results
    
    def send_performance_report(self, performance_data):
        """
        Send a performance report
        
        Args:
            performance_data (dict): Dictionary with performance metrics
            
        Returns:
            dict: Dictionary with results for each channel
        """
        # Format the performance report
        subject = f"Trading Performance Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        message = f"*Trading Performance Report*\n\n"
        message += f"*Date:* {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        message += "*Performance Metrics:*\n"
        for key, value in performance_data.items():
            message += f"*{key}:* {value}\n"
        
        # Send the report
        return self.send_alert(message, subject)

def configure_alerts():
    """
    Configure alerts interactively
    """
    alert_manager = AlertManager()
    
    # Configure Telegram
    print("\n=== Telegram Configuration ===")
    enable_telegram = input("Enable Telegram alerts? (y/n): ").lower() == 'y'
    alert_manager.config['telegram']['enabled'] = enable_telegram
    
    if enable_telegram:
        token = input("Enter Telegram bot token: ")
        chat_id = input("Enter Telegram chat ID: ")
        
        alert_manager.config['telegram']['token'] = token
        alert_manager.config['telegram']['chat_id'] = chat_id
    
    # Configure Slack
    print("\n=== Slack Configuration ===")
    enable_slack = input("Enable Slack alerts? (y/n): ").lower() == 'y'
    alert_manager.config['slack']['enabled'] = enable_slack
    
    if enable_slack:
        webhook_url = input("Enter Slack webhook URL: ")
        alert_manager.config['slack']['webhook_url'] = webhook_url
    
    # Configure Email
    print("\n=== Email Configuration ===")
    enable_email = input("Enable Email alerts? (y/n): ").lower() == 'y'
    alert_manager.config['email']['enabled'] = enable_email
    
    if enable_email:
        smtp_server = input("Enter SMTP server (default: smtp.gmail.com): ") or "smtp.gmail.com"
        smtp_port = int(input("Enter SMTP port (default: 587): ") or "587")
        username = input("Enter email username: ")
        password = input("Enter email password: ")
        from_email = input("Enter sender email: ")
        to_email = input("Enter recipient email: ")
        
        alert_manager.config['email']['smtp_server'] = smtp_server
        alert_manager.config['email']['smtp_port'] = smtp_port
        alert_manager.config['email']['username'] = username
        alert_manager.config['email']['password'] = password
        alert_manager.config['email']['from_email'] = from_email
        alert_manager.config['email']['to_email'] = to_email
    
    # Save configuration
    alert_manager.save_config()
    
    print("\nConfiguration saved. Testing alerts...")
    
    # Test alerts
    test_message = "This is a test alert from the trading system."
    results = alert_manager.send_alert(test_message, "Test Alert")
    
    print("\nAlert test results:")
    for channel, success in results.items():
        print(f"{channel}: {'Success' if success else 'Failed'}")

def main():
    """
    Main function to run the alerts module
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage trading alerts')
    parser.add_argument('--configure', action='store_true', help='Configure alerts')
    parser.add_argument('--test', action='store_true', help='Test alerts')
    parser.add_argument('--message', type=str, help='Message to send')
    parser.add_argument('--subject', type=str, help='Email subject')
    parser.add_argument('--channel', type=str, choices=['telegram', 'slack', 'email', 'all'], default='all', help='Channel to send to')
    args = parser.parse_args()
    
    alert_manager = AlertManager()
    
    if args.configure:
        configure_alerts()
    elif args.test:
        test_message = "This is a test alert from the trading system."
        channels = ['telegram', 'slack', 'email'] if args.channel == 'all' else [args.channel]
        results = alert_manager.send_alert(test_message, "Test Alert", channels)
        
        print("\nAlert test results:")
        for channel, success in results.items():
            print(f"{channel}: {'Success' if success else 'Failed'}")
    elif args.message:
        channels = ['telegram', 'slack', 'email'] if args.channel == 'all' else [args.channel]
        results = alert_manager.send_alert(args.message, args.subject, channels)
        
        print("\nAlert results:")
        for channel, success in results.items():
            print(f"{channel}: {'Success' if success else 'Failed'}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
