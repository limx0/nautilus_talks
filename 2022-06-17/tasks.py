from invoke import task


@task
def extract_catalog(c):
    c.run("tar xf ./demo/data/catalog.tar.bz2 --directory=demo/catalog")


@task
def slideshow(c):
    c.run("poetry run voila 20220518.ipynb --template reveal")
