# Build the React Native package
cd react 

# Remove directories if they still exist
[ -d ios ] && rm -rf ios
[ -d android ] && rm -rf android
[ -d node_modules ] && rm -rf node_modules

# Copy the ios and android directories
cp -R ../ios/* ios
cp -R ../android android

yarn 
yarn build