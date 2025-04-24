# Build the React Native package
cd react 

# Remove existing directories if they exist
rm -rf ios android

# Copy the ios and android directories
cp -R ../ios ios
cp -R ../android android

yarn 
yarn build