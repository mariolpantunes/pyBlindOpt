.PHONY: docs

docs:
	# 1. Clean previous build
	rm -rf docs

	# 2. Generate HTML (linking to the assets folder)
	pdoc --math -d google -o docs src/pyBlindOpt \
		--logo "assets/pyblindopt_logo.svg" \
		--favicon "assets/pyblindopt_logo.svg"

	# 3. Manually copy the assets folder into docs so the link works
	cp -r assets docs/assets

	@echo "Documentation built successfully!"