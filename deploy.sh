#!/bin/bash

# WTI Crude Oil Trading Bot Deployment Script
# This script deploys the trading bot to a cloud server

# Exit on error
set -e

# Configuration
REMOTE_USER=${REMOTE_USER:-"ubuntu"}
REMOTE_HOST=${REMOTE_HOST:-"your-server-ip"}
REMOTE_DIR=${REMOTE_DIR:-"/home/ubuntu/trading-bot"}
SSH_KEY=${SSH_KEY:-"~/.ssh/id_rsa"}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  WTI Crude Oil Trading Bot Deployment   ${NC}"
echo -e "${GREEN}=========================================${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo -e "${YELLOW}Please create a .env file with your configuration.${NC}"
    echo -e "${YELLOW}You can use .env.example as a template.${NC}"
    exit 1
fi

# Check if SSH key exists
if [ ! -f $(eval echo $SSH_KEY) ]; then
    echo -e "${RED}Error: SSH key not found at $SSH_KEY!${NC}"
    echo -e "${YELLOW}Please provide a valid SSH key path using SSH_KEY environment variable.${NC}"
    exit 1
fi

# Build Docker image locally
echo -e "\n${GREEN}Building Docker image...${NC}"
docker-compose build

# Create remote directory if it doesn't exist
echo -e "\n${GREEN}Creating remote directory...${NC}"
ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Copy files to remote server
echo -e "\n${GREEN}Copying files to remote server...${NC}"
rsync -avz --exclude 'venv' --exclude '.git' --exclude '__pycache__' \
    --exclude 'data/*.csv' --exclude 'models/*.pkl' --exclude 'logs/*.log' \
    -e "ssh -i $SSH_KEY" ./ $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/

# Copy .env file separately
echo -e "\n${GREEN}Copying .env file...${NC}"
scp -i $SSH_KEY .env $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/

# Deploy on remote server
echo -e "\n${GREEN}Deploying on remote server...${NC}"
ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_DIR && docker-compose up -d"

# Check deployment status
echo -e "\n${GREEN}Checking deployment status...${NC}"
ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_DIR && docker-compose ps"

echo -e "\n${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "${YELLOW}To check logs:${NC}"
echo -e "ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST \"cd $REMOTE_DIR && docker-compose logs -f\""
echo -e "${GREEN}=========================================${NC}"
