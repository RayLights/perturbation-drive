# Important Note: Git History Size

## Current State
After removing large binary files from the repository, the working directory is now much smaller (~8MB of actual code and assets), but the total repository size is still ~505MB due to the .git history folder (497MB).

## Why the .git Folder is Still Large
Git stores the complete history of all changes. Even though we removed the large files from the current branch, they still exist in the git history. This means:
- Fresh clones still download the entire history (~500MB)
- The large files are still in `.git/objects/`
- The repository size hasn't been fully reduced yet

## To Fully Reduce Repository Size (Optional)
To completely remove large files from git history and achieve a truly smaller repository, you would need to rewrite the git history. This is a more invasive operation that affects all branches and requires coordination with all contributors.

### Option 1: Using git filter-repo (Recommended)
```bash
# Install git-filter-repo
pip install git-filter-repo

# Backup your repository first!
git clone --mirror <your-repo-url> backup.git

# Filter out the large files/directories from history
git filter-repo --path examples/udacity/sim/udacity_linux/ --invert-paths
git filter-repo --path examples/self_driving_sandbox_donkey/sim/sdsim_linux/ --invert-paths
git filter-repo --path perturbationdrive/OverlayMasks/MasksZip.zip --invert-paths
git filter-repo --path checkpoints/original_extended3_extended5.h5 --invert-paths
# ... repeat for other large files

# Force push to update the remote
git push origin --force --all
git push origin --force --tags
```

### Option 2: Using BFG Repo-Cleaner
```bash
# Download BFG
# https://rtyley.github.io/bfg-repo-cleaner/

# Remove large files
java -jar bfg.jar --delete-files "*.h5" your-repo.git
java -jar bfg.jar --delete-files "*.mp4" your-repo.git
java -jar bfg.jar --delete-folders "{udacity_linux,sdsim_linux}" your-repo.git

# Clean up
cd your-repo.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push
git push origin --force --all
```

### Option 3: Start Fresh (Simplest but Loses History)
If git history is not critical, you could:
1. Create a new repository
2. Copy only the current files (without .git)
3. Initialize fresh git history
4. This would give you a repository of ~8MB instead of ~500MB

## Important Warnings
⚠️ **Rewriting history is a destructive operation**:
- All collaborators need to re-clone the repository
- Open pull requests may need to be recreated
- Any local clones will be out of sync
- Backup everything before proceeding
- Coordinate with your team

## Recommended Approach
For this repository, we recommend:
1. **Keep the current changes** (files removed from HEAD)
2. **Document** where to download large assets (already done)
3. **Consider history rewrite** only if the repository size is causing significant issues
4. **If rewriting history**, do it as a separate, coordinated effort with all contributors

## Current Benefits (Without History Rewrite)
Even without rewriting history, the current changes provide benefits:
- ✅ New clones get smaller working directory (~8MB vs ~800MB)
- ✅ Faster git operations on the working tree
- ✅ Less storage needed for working copies
- ✅ Faster CI/CD pipelines that only need current files
- ✅ Clear documentation on where to get large assets

## Future Clones
When someone clones the repository after these changes:
- They download ~500MB (git history)
- Working directory is only ~8MB
- They can download large assets separately as needed
- Much faster to work with on a daily basis

If you decide to rewrite history later, the reduction would be more dramatic (500MB → ~10-20MB total).
