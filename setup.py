from setuptools import setup, find_packages

setup(
    name='needlehaystack',
    version='0.1.0',
    author='dff652',
    author_email='wcdzdff@gmail.com',
    description='Use needlehaystack method to test local llms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dff652/LLM_Test',
    packages=find_packages(),
    include_package_data=True,  # This includes non-code files specified in MANIFEST.in
    install_requires=[
        x for x in open("./requirements.txt", "r+").readlines() if x.strip()
    ],
    python_requires='>=3.6',
    classifiers=[],
    entry_points={
        'console_scripts': [
            'needlehaystack.run_test = needlehaystack.run:main',
        ],
    },
)
