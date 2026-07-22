async def fetch_url(
    client,
    url
):


    if url in SCRAPER_CACHE:

        return SCRAPER_CACHE[url]


    async with REQUEST_LIMIT:

        try:

            response = await client.get(url)


            response.raise_for_status()


            content_type = response.headers.get(
                "content-type",
                ""
            )


            if "pdf" in content_type:


                data = response.content


            else:

                data=response.text



            SCRAPER_CACHE[url]=data


            return data



        except Exception as e:


            logger.warning(
                f"Fetch failed {url}: {e}"
            )


            return None
