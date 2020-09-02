# ge_data_manager

A dockerized django/postgresql front-end to view and manage PyGEST results

If you have run PyGEST push on a bunch of matrices, you'll have a lot of results files sitting around. This project creates a django app within a docker container that can organize those results for you.

## Prerequisites

You have docker and docker-compose installed.

## Installation

Run the following from the linux command line.

    git clone https://github.com/mfschmidt/ge_data_manager.git
    cd ge_data_manager
   
### Create a new empty database (should automate this)

    docker run --name some-postgres -e POSTGRES_PASSWORD=postgres -v /home/mike/Projects/ge_data_manager/pgdata:/var/lib/postgresql/data postgres

or

    # sudo -u postgres psql
    postgres=# CREATE DATABASE _;
    postgres=# \q;

### Provide ge_data_manager access to your PyGEST data

Either (Any <b>one</b> of these will allow the docker container to find your files.)
1. Create a link to your data at <code>/var/ge_data/</code> with a command like <code>$ sudo ln -s /home/mike/ge_data /var/ge_data</code>,
2. Copy all of your data to <code>/var/ge_data/</code>, or
3. Edit <code>docker-compose.yml</code> to map your own <code>PYGEST_DATA</code> path to <code>/data</code>.

Then run the containers.

    docker-compose build && docker-compose up
    
You can now browse to <code> http://localhost:8000 </code> to explore ge_data_manager.

## Optional steps

This will run on 'localhost' or '127.0.0.1' by default, but will be accessible from any address on your network. Working alone behind a firewall, no problem. But if you want to run this site while on a larger network or open to a public internet, you should change two variables in settings.py.

    # ORIGINAL, FOR DEBUGGING AND TESTING - ANYONE CAN ACCESS THE SITE AND GET DEBUG INFORMATION
    DEBUG = True
    ALLOWED_HOSTS = ['*']
    
    # NEW, SAFER FOR USING YOURSELF AND SHARING WITH YOUR COLLEAGUES AT 192.168.1.2 AND 192.168.1.3
    DEBUG = False
    ALLOWED_HOSTS = ['localhost', '192.168.1.2', '192.168.1.3']
    
You can add yourself as a superuser, although you don't need it for anything. This also demonstrates how to access any django <code>manage.py</code> functions within the container. First, get the name of the docker container you've just started. It's usually the same, but can vary.

    docker ps
    
Use the name of the <code>gedatamanager_web</code> docker container you just started, and execute an interactive bash shell inside of it.

    docker exec -it gedatamanager_web_1 /bin/bash
    
or
    
    docker-compose run web /bin/bash

And within that shell, in the context of the docker container, you can interact with the django sub-system.

    python manage.py createsuperuser
    python manage.py shell
    python manage.py migrate

If you created a new superuser, you can log in at http://localhost:8000/admin/ to play around with it. The user you created will only be available on your local instance of ge_data_manager. If you clone it elsewhere, you'll get a fresh new empty database with no users and no results.

## Some debugging help

If you'd like to monitor celery tasks as they run, you can change the worker's command in docker-compose.yml to `--loglevel=debug`. This will emit more detailed logs from the worker container.

If you make significant changes to the models, you may need to manually migrate those changes to the db structure.
You can accomplish this as follows:

    docker-compose run web bash
    python manage.py makemigrations
    python manage.py migrate

## Additional info, or what ge_data_manager does

1. calculate_individual_stats (will not overwrite cache)
    for each tsv in the queryset:
        store results_as_dict(tsv) as json
    store individual summaries as one cached dataframe

2. calculate_group_stats (will not overwrite cache)
    for each shuffle-type:
        calculate % similarity and kendall tau
        for each split:
            calculate % similarity and kendall tau
        for each seed:
            calculate % similarity and kendall tau
        calculate real-vs-shuffle % similarity and kendall tau
    store group summary as one cached dataframe

3. Build plots and gene lists.

